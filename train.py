#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import csv
from lpipsPyTorch import lpips
from einops import reduce, repeat

if False: 
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
else: 
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    opt.position_lr_max_steps = opt.iterations

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):      
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()


        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        model_id = torch.randint(0, gaussians.n_models, (1,)).item()
        gaussians.model_id = model_id

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        loss_kl_scal = gaussians.compute_kl_uniform_scal()
        loss_kl_xyz = gaussians.compute_kl_xyz()
        loss_kl_opacity = gaussians.compute_kl_opacity()

        loss += 1.0*(loss_kl_scal + loss_kl_xyz + loss_kl_opacity)

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()


        loss.backward()

        iter_end.record()

        if iteration == opt.iterations:
            for param in gaussians.parameters():
                param.requires_grad_(True)

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            spawn_interval = dataset.spawn_interval

            # Log and save
            training_report(dataset, tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration % spawn_interval == 0:  # spawn interval should be a multiple of densification interval
                    gaussians.spawn(scene.cameras_extent)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None

                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

import torchvision

def training_report(dataset, tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()
        if iteration == testing_iterations[-1]: 
            render_set(dataset, scene, renderArgs[0])

from utils.image_utils import psnr, nll_kernel_density, ause_br
from gaussian_renderer import render, forward_k_times
from os import makedirs


def render_set(dataset, scene, pipeline):
    gaussians, views = scene.gaussians, scene.getTestCameras()

    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    psnr_all, ssim_all, lpips_all, ause_mae_all, mean_nll_all, depth_ause_mae_all = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    eval_depth = True if dataset.dataset_name == "LF" else False

    scene_name = scene.model_path.split("/")[-1]

    render_path = f"{scene.model_path}/test/ours_7000/renders"
    gts_path = f"{scene.model_path}/test/ours_7000/gt"
    unc_path = f"{scene.model_path}/test/ours_7000/unc"
    fisher_unc_path = f"{scene.model_path}/test/ours_7000/fisher_unc"

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(unc_path, exist_ok=True)
    makedirs(fisher_unc_path, exist_ok=True)

    # FisherRF uncertainty setup
    params = [
        gaussians._xyz,
        gaussians._features_dc,
        gaussians._features_rest,
        gaussians._scaling,
        gaussians._rotation,
        gaussians._opacity
    ]
    name2idx = {"xyz": 0, "rgb": 1, "sh": 2, "scale": 3, "rotation": 4, "opacity": 5}
    filter_out_idx = [name2idx[k] for k in ["rotation"]]  # Match FisherRF’s default
    params = [p.requires_grad_(True) for i, p in enumerate(params) if i not in filter_out_idx]
    optim = torch.optim.SGD(params, 0.)
    gaussians.optimizer = optim

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        gt = view.original_image[0:3, :, :]
        out = forward_k_times(view, gaussians, pipeline, background)
        mean = out['comp_rgb'].detach()
        rgbs = out['comp_rgbs'].detach()
        std = out['comp_std'].detach()
        depths = out['depths'].detach()
        pixel_gaussian_counter = out['pixel_gaussian_counter'].detach()

        # FisherRF uncertainty
        render_pkg = render(view, gaussians, pipeline, background)
        pred_img = render_pkg["render"]  # Should have grad_fn
        pred_img.backward(gradient=torch.ones_like(pred_img))
        H_per_gaussian = sum(reduce(p.grad.detach(), "n ... -> n", "sum") for p in params)
        hessian_color = repeat(H_per_gaussian.detach(), "n -> n c", c=3)
        to_homo = lambda x: torch.cat([x, torch.ones(x.shape[:-1] + (1,), dtype=x.dtype, device=x.device)], dim=-1)
        pts3d_homo = to_homo(gaussians._xyz)
        pts3d_cam = pts3d_homo @ view.world_view_transform
        gaussian_depths = pts3d_cam[:, 2, None]
        hessian_color = hessian_color * gaussian_depths.clamp(min=0)
        render_pkg_unc = render(view, gaussians, pipeline, background, override_color=hessian_color)
        fisher_uncertainty = reduce(render_pkg_unc["render"], "c h w -> h w", "mean")
        fisher_uncertainty_map = torch.log(fisher_uncertainty / pixel_gaussian_counter).clamp(min=0).detach()
        optim.zero_grad(set_to_none=True)

        if eval_depth: 
            depths = depths * scene.depth_scale

            depth = depths.mean(dim=0)
            depth_std = depths.std(dim=0)
            depth_gt = view.depth

            depth_mae = ((depth - depth_gt)).abs()
            depth_ause_mae, depth_ause_err_mae, depth_ause_err_by_var_mae = ause_br(depth_std.reshape(-1), depth_mae.reshape(-1), err_type='mae')
            depth_ause_mae_all += depth_ause_mae


        unc_vis_multiply = 10
        torchvision.utils.save_image(mean, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(unc_vis_multiply*std, os.path.join(unc_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(fisher_uncertainty_map, os.path.join(fisher_unc_path, f'{idx:05d}.png'))


    psnr_all /= len(views)
    ause_mae_all /= len(views)
    mean_nll_all /= len(views)
    ssim_all /= len(views)
    lpips_all /= len(views)

    depth_ause_mae_all /= len(views)

    csv_file = f"output/eval_results_{dataset.dataset_name}.csv"
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)

        if eval_depth: 
            results = f"\nEvaluation Results: PSNR {psnr_all} SSIM {ssim_all} LPIPS {lpips_all} AUSE {ause_mae_all} NLL {mean_nll_all} Depth AUSE {depth_ause_mae_all}"
            print(results)
            writer.writerow([dataset.dataset_name, scene_name, psnr_all, ssim_all, lpips_all, ause_mae_all, mean_nll_all, depth_ause_mae_all])
        else: 
            results = f"\nEvaluation Results: PSNR {psnr_all} SSIM {ssim_all} LPIPS {lpips_all} AUSE {ause_mae_all} NLL {mean_nll_all}"
            print(results)
            writer.writerow([dataset.dataset_name, scene_name, psnr_all, ssim_all, lpips_all, ause_mae_all, mean_nll_all])

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3_000, 7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[3_000, 7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])

    args.test_iterations.append(args.iterations)
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    #network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")

    print("\nTesting gradient propagation:")
    gaussians = GaussianModel(lp.extract(args).sh_degree)
    scene = Scene(lp.extract(args), gaussians)
    test_view = scene.getTestCameras()[0]
    params = [gaussians._xyz, gaussians._features_dc, gaussians._features_rest, gaussians._scaling, gaussians._opacity]
    for p in params:
        p.requires_grad_(True)
    render_pkg = render(test_view, gaussians, pp.extract(args), background)
    pred_img = render_pkg["render"]
    print(f"Test pred_img requires_grad: {pred_img.requires_grad}, grad_fn: {pred_img.grad_fn}")
    pred_img.backward(gradient=torch.ones_like(pred_img))
    print("Gradient test passed.")
