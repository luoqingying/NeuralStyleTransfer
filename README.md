# NeuralStyleTransfer

This project reproduces the procedure described in Gatys et al., “Image Style Transfer Using Convolutional Neural Networks,” (https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_p aper.pdf).

The user input will require path for content and style image, value of alpha and beta, output image size, and number of iteration. For other parameters, like which layer to optimize, which optimizer and loss function to use, and etc, should be manually modified inside the code.

NST.py includes all code you need to run the program. The output image will be stored at current directory with name 'output.jpg'. content and style folder contains some image I used when I experimented with thie network.

I also attached the report for detailed implementation and poster for better visualization purpose.
