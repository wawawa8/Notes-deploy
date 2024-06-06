---
{"dg-publish":true,"permalink":"/Course Notes/Raytracing in one weekend/Notes/","created":"2024-02-26T00:02:01.713+08:00"}
---

# coordinate system

![Pasted image 20240120144722.png|450](/img/user/Course%20Notes/Raytracing%20in%20one%20weekend/assets/Pasted%20image%2020240120144722.png)

# surface normal

- use unit-length normal everywhere
- when calculating normals, try best to avoid time-consuming `sqrt` operator: try find the length from other places

# ray intersection

## with sphere

we need to solve a quadratic equation.
![Pasted image 20240120151949.png|725](/img/user/Course%20Notes/Raytracing%20in%20one%20weekend/assets/Pasted%20image%2020240120151949.png)
use half-vector to improve efficiency!

## shadow acne(粉刺, /ˈækni/)

the calculated intersection point might be inside the surface

If we start next reflection directly from the calculated position, there is problem of acne (lots of black dots)

start from $pos + 0.0001 direction$

## Lambertian reflection

diffuse reflection:
    a diffuse reflected ray is most likely to scatter near the surface normal

to model lambertian reflection:

![Pasted image 20240121135414.png|450](/img/user/Course%20Notes/Raytracing%20in%20one%20weekend/assets/Pasted%20image%2020240121135414.png)

uniformly sample a unit vector and add it to normal $\boldsymbol{n}$

 