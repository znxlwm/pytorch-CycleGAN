# pytorch-CycleGAN
Pytorch implementation of CycleGAN [1].

* you can download datasets: https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/
* you can see more information for network architecture and training details in https://arxiv.org/pdf/1703.10593.pdf

## dataset
* apple2orange
  * apple training images: 995, orange training images: 1,019, apple test images: 266, orange test images: 248
* horse2zebra
  * horse training images: 1,067, zebra training images: 1,334, horse test images: 120, zebra test images: 140

## Resutls
### apple2orange (after 200 epochs)
* apple2orange
<table align='center'>
<tr align='center'>
<td> Input </td>
<td> Output </td>
<td> Reconstruction </td>
</tr>
<tr>
<td><img src = 'images/apple2orange/AtoB/1_input.png'>
<td><img src = 'images/apple2orange/AtoB/1_output.png'>
<td><img src = 'images/apple2orange/AtoB/1_recon.png'>
</tr>
<tr>
<td><img src = 'images/apple2orange/AtoB/2_input.png'>
<td><img src = 'images/apple2orange/AtoB/2_output.png'>
<td><img src = 'images/apple2orange/AtoB/2_recon.png'>
</tr>
<tr>
<td><img src = 'images/apple2orange/AtoB/3_input.png'>
<td><img src = 'images/apple2orange/AtoB/3_output.png'>
<td><img src = 'images/apple2orange/AtoB/3_recon.png'>
</tr>
<tr>
<td><img src = 'images/apple2orange/AtoB/4_input.png'>
<td><img src = 'images/apple2orange/AtoB/4_output.png'>
<td><img src = 'images/apple2orange/AtoB/4_recon.png'>
</tr>
<tr>
<td><img src = 'images/apple2orange/AtoB/5_input.png'>
<td><img src = 'images/apple2orange/AtoB/5_output.png'>
<td><img src = 'images/apple2orange/AtoB/5_recon.png'>
</tr>
</table>

* orange2apple
<table align='center'>
<tr align='center'>
<td> Input </td>
<td> Output </td>
<td> Reconstruction </td>
</tr>
<tr>
<td><img src = 'images/apple2orange/BtoA/1_input.png'>
<td><img src = 'images/apple2orange/BtoA/1_output.png'>
<td><img src = 'images/apple2orange/BtoA/1_recon.png'>
</tr>
<tr>
<td><img src = 'images/apple2orange/BtoA/2_input.png'>
<td><img src = 'images/apple2orange/BtoA/2_output.png'>
<td><img src = 'images/apple2orange/BtoA/2_recon.png'>
</tr>
<tr>
<td><img src = 'images/apple2orange/BtoA/3_input.png'>
<td><img src = 'images/apple2orange/BtoA/3_output.png'>
<td><img src = 'images/apple2orange/BtoA/3_recon.png'>
</tr>
<tr>
<td><img src = 'images/apple2orange/BtoA/4_input.png'>
<td><img src = 'images/apple2orange/BtoA/4_output.png'>
<td><img src = 'images/apple2orange/BtoA/4_recon.png'>
</tr>
<tr>
<td><img src = 'images/apple2orange/BtoA/5_input.png'>
<td><img src = 'images/apple2orange/BtoA/5_output.png'>
<td><img src = 'images/apple2orange/BtoA/5_recon.png'>
</tr>
</table>

* Learning Time
  * apple2orange - Avg. per epoch: 299.38 sec; Total 200 epochs: 62,225.33 sec

### horse2zebra (after 200 epochs)
* horse2zebra
<table align='center'>
<tr align='center'>
<td> Input </td>
<td> Output </td>
<td> Reconstruction </td>
</tr>
<tr>
<td><img src = 'images/horse2zebra/AtoB/1_input.png'>
<td><img src = 'images/horse2zebra/AtoB/1_output.png'>
<td><img src = 'images/horse2zebra/AtoB/1_recon.png'>
</tr>
<tr>
<td><img src = 'images/horse2zebra/AtoB/2_input.png'>
<td><img src = 'images/horse2zebra/AtoB/2_output.png'>
<td><img src = 'images/horse2zebra/AtoB/2_recon.png'>
</tr>
<tr>
<td><img src = 'images/horse2zebra/AtoB/3_input.png'>
<td><img src = 'images/horse2zebra/AtoB/3_output.png'>
<td><img src = 'images/horse2zebra/AtoB/3_recon.png'>
</tr>
<tr>
<td><img src = 'images/horse2zebra/AtoB/4_input.png'>
<td><img src = 'images/horse2zebra/AtoB/4_output.png'>
<td><img src = 'images/horse2zebra/AtoB/4_recon.png'>
</tr>
<tr>
<td><img src = 'images/horse2zebra/AtoB/5_input.png'>
<td><img src = 'images/horse2zebra/AtoB/5_output.png'>
<td><img src = 'images/horse2zebra/AtoB/5_recon.png'>
</tr>
</table>

* zebra2horse
<table align='center'>
<tr align='center'>
<td> Input </td>
<td> Output </td>
<td> Reconstruction </td>
</tr>
<tr>
<td><img src = 'images/horse2zebra/BtoA/1_input.png'>
<td><img src = 'images/horse2zebra/BtoA/1_output.png'>
<td><img src = 'images/horse2zebra/BtoA/1_recon.png'>
</tr>
<tr>
<td><img src = 'images/horse2zebra/BtoA/2_input.png'>
<td><img src = 'images/horse2zebra/BtoA/2_output.png'>
<td><img src = 'images/horse2zebra/BtoA/2_recon.png'>
</tr>
<tr>
<td><img src = 'images/horse2zebra/BtoA/3_input.png'>
<td><img src = 'images/horse2zebra/BtoA/3_output.png'>
<td><img src = 'images/horse2zebra/BtoA/3_recon.png'>
</tr>
<tr>
<td><img src = 'images/horse2zebra/BtoA/4_input.png'>
<td><img src = 'images/horse2zebra/BtoA/4_output.png'>
<td><img src = 'images/horse2zebra/BtoA/4_recon.png'>
</tr>
<tr>
<td><img src = 'images/horse2zebra/BtoA/5_input.png'>
<td><img src = 'images/horse2zebra/BtoA/5_output.png'>
<td><img src = 'images/horse2zebra/BtoA/5_recon.png'>
</tr>
</table>

* Learning Time
  * horse2zebra - Avg. per epoch: 299.25 sec; Total 200 epochs: 61,221.27 sec

## Development Environment

* Ubuntu 14.04 LTS
* NVIDIA GTX 1080 ti
* cuda 8.0
* Python 2.7.6
* pytorch 0.1.12
* matplotlib 1.3.1
* scipy 0.19.1

## Reference

[1] Zhu, Jun-Yan, et al. "Unpaired image-to-image translation using cycle-consistent adversarial networks." arXiv preprint arXiv:1703.10593 (2017).

(Full paper: https://arxiv.org/pdf/1703.10593.pdf)
