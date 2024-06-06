---
{"dg-publish":true,"permalink":"/Project Notes/Watertight manifold mesh reconstruction with inverse rendering/Watertight manifold mesh reconstruction with inverse rendering/","created":"2024-02-26T00:01:53.876+08:00"}
---

- 2024.1.3
    - stage3
        - given: stage2 output
            - mesh
            - appearance network
                - albedo
                - roughness
            - light
        - 1. load in mesh
    		- load geometry network
    		- transform to mesh
    		- **first use gt mesh**
    - stage2
        - **first use gt mesh to train stage2**
        - 
    - problems
        - hotdog-tensoir use which mesh?
            - validate PSNR: 34.320 using DMTet and old version of training code
            - 34.775 using DiffMC
- 2024.1.6
    - current experiment based on hotdog.obj, **need to redo with hotdog_triangle.obj**
- 2024.1.11
    - why need `dx_du` and `dy_dv` 
        - to get multiple rays with small change in direction for one single pixel
        - get dx_du and dy_dv
            - FIPT:
                - c2w = transform_matrix[:3, :4]
                - R = c2w[:, :3]
                - directions = torch.stack([-(i-W/2)/focal, -(j-H/2)/focal, torch.ones_like(i)], -1) 
                - rays_d = directions @ R.T
                - dxdu = torch.tensor([1.0/focal,0,0])[None,None].expand_as(directions)@R.T
                - dydv = torch.tensor([0,1.0/focal,0])[None,None].expand_as(directions)@R.T
            - Ours:
                - pose = transform_matrix @ blender2opencv
                - c2w = pose
                - directions = torch.stack([(x - center[0]) / focal[0], (y - center[1]) / focal[1], torch.ones_like(x)], -1) /= torch.norm(self.directions)
                - rays_d = directions @ c2w[:3, :3].T
    - get_ray_intersection
        - `ray_intersect` in path_tracing.py
            - return
                - positions
                - normals
                - uvs: uv coordinates
                - idx: triangle indices
                - valid: [B], whether valid
- 2024.1.15
    - `def path_tracing`
- 2024.1.17
    - `def path_tracing`
        - scene: mitsuba scene
        - rays_o: Bx3 ray origin (camera pose)
        - rays_d: Bx3 ray direction (camera -> )
        - dx_du,dy_dv: Bx3 ray differential ($1/focal$)
            - used for multiple samples on a single pixel
        - spp: samples per pixel
        - indir_depth: indirect illumination depth
        - 
        - du, dv: [2, B, spp, 1] in [-0.5, 0.5]
        - wi: rays_d + dx_du * du + dy_dv * dv [2, B, spp, 3]->[nB, 3]
            - $(d_x, d_y, d_z) + (1/f, 0, 0) * \text{spp个随机的[-0.5,0.5] 的数}$
            - 原始的相邻两个像素的 ray_d 只相差 $1/f$，所以随机的范围在 $-0.5/f\sim0.5/f$
            - 新的做完 spp 个 sample 之后的入射方向
        - position: rays_o, starter position, [nB, 3]
        - do ray_intersect
        - position: intersection positions
        - normal: [nB, 3]
        - \_: ret.uv, uv positions
        - triangle_idx
        - vis: whether valid intersections
        - 
        - L, \_, valid_next = *eval_emitter*(position, wi, triangle_idx)
        - (we don't have emitter)
            - vis: has mesh, triangle_idx != -1
            - is_area: both has mesh & is emitter (we don't have these areas)
            - Le: [nB, 3]
            - emit_pdf: [nB,]
            - For those is_area, find the corresponding light,
                - emit_pdf = emitter_pdf / emitter_area
                - Le = radiance of this light
            - valid_next = those has mesh & is not an emitter
        - position = position[valid_next], normal = normal[valid_next], wo = -wi[valid_next] (size [B2])
        - active_next = valid_next.clone()
        - mat = material_net(position)
            - query albedo, roughness and metallic
        - `sample_emitter`
            - sample1: [B2], uniform sampled in [0, 1]
            - sample2: [B2, 2], uniform sampled in [0, 1]
            - position
            - first sample B2 emitters according to the cdf function
            - sample one emitter for each position,
            - wi denotes the direction from position -> emitter [B2, 3]
            - pdf = emitter_pdf / emiiter_area 单位面积的 pdf
            - traingle_idx = the triangle index of the emitter position [B2,]
        - `do another ray_intersect`
            - scene
            - position + wi * $\epsilon$ 
            - wi
            - return: emit_position, emit_normal, triangle_idx, emit_valid
            - let the position go to the sampled emitter direction
        - emit_vis = can connect to the sampled emitter or 直接不与任何东西相交进空气 （一次反射直接结束）
            - whether ray doesn't intersect (intersect with background) or can go to the emitter
        - emit_weight: radiance of the sampled emitter
        - G = (-wi * emit_normal)... 
            - 根据光源的 normal 和光源到 postion 的距离，计算光源对这个 position 的强度
            - 把那些不会遇到光源的地方的 G 设为 1
        - emit_weight = emit_weight * emit_vis * G / emit_pdf 
            - radiance of the emitter * whether can connect to this position * 光源关于这个点的强度 / 单位面积pdf
        - emit_brdf, brdf_pdf 根据 albedo, roughness, metallic 以及入射出射方向 以及 normal 计算 brdf 和 pdf
            - brdf = brdf_diffuse + brdf_specular
            - pdf = 0.5 * pdf_spec + 0.5 * pdf_diff
        - brdf_pdf \*= G
            - pdf 感觉就是整个出射光分在 wo 方向上的 概率
        - L[active_next] += emit_brdf \* emit_weight \* ${emit\_pdf ^ 2} / {(emit\_pdf^2 + brdf\_pdf^2)}$
            - emitter 的 radiance \* emit_brdf \* ???
        - `sample_brdf`
            - sample1: [B2,]
            - sample2: [B2, 2]
            - wo, normal, mat
                - wo 是最开始的入射方向的反方向，也就是从 position 指向 camera 的方向，也是 viewing direction
            - mask = (sample1 > 0.5)
            - wi[mask] = diffuse_sampler
            - wi[~mask] = specular_sampler
            - 感觉就是选择下一次光的方向
            - brdf, pdf = self.eval_brdf(wi, wo, normal, mat)
            - brdf_weight = brdf/pdf
        - 得到了 wi, brdf_pdf, brdf_weight
        - ray_intersect 找下一次交点
            - position_next
            - normal
            - triangle_idx
            - vis
        - mat_next 计算下一次交点的 brdf
        - 如果下一次交点有 emitter?
            - 我们不用管
        - wo = -wi，接着做下一次反射
        - position = position_next
        - 去掉下一次交点是 emitter 的那些
        - position = position[valid_next], wo = wo[valid_next], normal = normal[valid_next], brdf_weight = brdf_weight[valid_next]
        - why disable gradient after 1st bounce?


- 2024.2.2
    - derived_normal: 根据 density field 计算出的 normal
    - predicted_normal: 根据 intrinsic feature 过 mlp 出的 normal
    - face_normal: 根据 mesh 本身出的 normal

- 2024.3.16
    - our own ray tracing
    - [[Project Notes/own ray tracing\|own ray tracing]]

- 2024.5.21
    - 设计实验

- 2024.6.5






my_render (Scene, material_1) = output
Blender (Scene, material_2) = output

设计 metric？
$$
\begin{gather}
I, I_0 \\
I'=I \times \frac {\text{mean}(I_0)} {\text{mean}(I)} \\
PSNR(I',I_0)
\end{gather}
$$
![Pasted image 20240606095049.png](/img/user/Project%20Notes/Watertight%20manifold%20mesh%20reconstruction%20with%20inverse%20rendering/assets/Pasted%20image%2020240606095049.png)



in nerfactor, scale each channel to minimize mse loss 看起来更不合理

![Pasted image 20240606095939.png](/img/user/Project%20Notes/Watertight%20manifold%20mesh%20reconstruction%20with%20inverse%20rendering/assets/Pasted%20image%2020240606095939.png)

![Pasted image 20240606100216.png](/img/user/Project%20Notes/Watertight%20manifold%20mesh%20reconstruction%20with%20inverse%20rendering/assets/Pasted%20image%2020240606100216.png)
