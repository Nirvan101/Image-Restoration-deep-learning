
# Image restoration by in-painting

Restoring images of damaged paintings using in-painting. Damaged paintings have discolored patches where the paint has faded or fallen off. These patches are often whitish. This project uses image in-painting to fill and restore these lost regions. 

This repository contains multiple models that I constructed to solve this task. The models include context-encoders, GANS, conditional GANS and pixel diffusion.

# Dataset

There are 68 images in the dataset provided. Out of these only 20 are of good quality. Hence, there are only 20 training images. Example of a good image:

[Training image](https://drive.google.com/open?id=1amYjXfVuwzYwTEStBai3L0t6NOSbi90K)

Example of a damaged image: 

[Test image](https://doc-14-8c-docs.googleusercontent.com/docs/securesc/5j15a90ban8ntvvgptfpmdg2b4ol3uat/pkel1tf057294q7caoeffdghk44kqr05/1546372800000/05158308479831370756/05158308479831370756/1POYPF4sYKwgPZHaK4xWz-ppEyiybFts_?e=view&nonce=5npqpfiualrp4&user=05158308479831370756&hash=eeq0o4ajlkuaq6s4nifm24k5hsch93sp)

As you can see, the damaged painting has many discolored patches which have become white. The aim is to use image in-painting to fill these white patches.


## Preprocessing:

The dataset is very small. The training dataset contains only 20 images. So, I have used data augmentation using the ImageDataGenerator. The images are all resized to (256,384,3) as this is the average image size in the dataset. They are then divided by 255 to normalize them. 

To models are trained on the good images. The image is first cropped artificially. This cropped version is input into the model and the original image is provided as the ground truth label. Hence, the model is effectively trained to convert the cropped images into their original forms.



## Models:

### Context Encoder

I constructed this model based on the paper 'Context Encoders: Feature Learning by Inpainting'  found here https://arxiv.org/pdf/1604.07379.pdf

This model contains an encoder and a decoder.  During training, the image is first cropped. A lot of tiny white holes are made on this image- these resemble the white patches that exist in damaged paintings. This modified image is fed into the encoder, it is downsampled into an encoding using Conv layers. The encoding is upsampled using Conv and Upsampling layers. The output is of the same size as the input i.e (256,384,3)
    
The original image is used as the ground truth label. I use a mean-square loss function and a sigmoid activation in the output layer. 

The output is then multiplied by 255 to get the final reconstructed output image.

----------------------------------------------------------------------------------------------------
 
### Context Encoder 2

This was my second design for the context encoder. Here I have added a Dense layer between the encoder and decoder to generate the encoding. The intuition is that the Dense layer will connect features from different regions of the image together and this will improve the inpainting performance.

-----------------------------------------------------------------------------------------------------

### Conditional GAN

I designed my conditional GAN based on the paper 'SEMI-SUPERVISED LEARNING WITH CONTEXT-CONDITIONAL GENERATIVE ADVERSARIAL NETWORK' found here https://arxiv.org/pdf/1611.06430v1.pdf

During training, from each image a white square is cropped out from the centre. This modified image is input into the generator.

The generator has an encoder-decoder network and it produces an image resembling the input image. This output image is passed to a custom keras layer which also receives the input image from the input layer. This custom layer replaces the masked central region of the input image with the corresponding central region of the generated image. This new image is the final output of the generator. The generator uses mean-square-error as it's loss function. This is callled reconstruction loss.

This image is passed to the discriminator. It predicts whether the image is original or generator-produced and this loss is called adversial loss. The use of adversial loss improves the training of the generator.

The difference between a regular GAN and a conditional GAN is the use of the custom layer after the generator. This trains the model to produce only the central masked region of the image (and not the entire image as was the case with GAN) using the surrounding regions.

-----------------------------------------------------------------------------------------------------

### GAN

This is a regular GAN where the generator has an mse loss function and the discriminator has a binary_cross_entropy loss function. The generator is designed to generate the entire input image back from the encoding. 

-----------------------------------------------------------------------------------------------------


### Pixel diff

This was a script I wrote using the opencv inpainting function. The user has to manually select a rectangular portion of the image and the script will automatically perform inpainting in that region.

