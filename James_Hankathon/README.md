## Languages and Frameworks Used 
  * Python: coding language
  * NumPy: library for numerical calculations
  * Matplotlib: library for data visualisation
  * Pytorch: a deep learning framework by Facebook AI Research Team for building neural networks
  * torchvision: package consists of popular datasets, model architectures, and common image transformations for computer vision
  * OpenCV: library for image processing
  * PyQt5: library for GUI programming and demonstration

## Setup
  After cloning, you can run the `main.py` file to see a quick demo of the project. In the search bar you can type "Earth" and view the different styles in our demo. Afterwards, you can run the `ImageProcessingExtensions.py` file to see a demo of basic image processing of imagery. 


# Code Review

  ## main.py
     ### **Usage**
     Go to the search bar on top and type **Earth**  &#8594; Hit the ** Go ** button  &#8594; now you can see the result of your search  &#8594; Press the ** Change style ** button to see a wide range of stylistic changes to the Earth image. 

    ### **Logic Flow**
    input the keywords &#8594;  downloads the first image queried in "https://images.nasa.gov/" to the assets folder  &#8594;  pass the image and a given style image to the StylizeImg.py file to train &#8594; result is a new image of the two images, content and style, combined and visualized  &#8594;  utilizes PyQt5 to implement an interactive GUI for more accessible and observable results


  ##NasaAPICrawler.py

    ### **Purpose**
    the backbone of our search engine. uses API provided by NASA to crawl imagery from NASA's vast dataset.  


  ###StylizeImg.py

    ### **Purpose**
    StylizeImg.py is an image combining algorithm that uses the VGG pretrained AI model to implement the merging of 2 images. the resulting image is a blend of the content of one of the images and the style of the other, similar to a customized style filter based on the image. 

    ### **Method**
    The main goal of project is to combine two images and produce new image. The combination works in slightly different way i.e., we combine the style of one image with the content of other image. First we take the image from which we want to extract content usually     called <b>content image</b> and take another image from which the style is to be extracted usually called **style image**. This is the implementation of [this](https://arxiv.org/pdf/1508.06576.pdf) research paper.  
  
  Convolutional Neural Networks are a type of neural networks which are used widely in Image classification and recongnition. A CNN architecture called VGG19 has been used in this project. The starting layers in this architecture extract the basic features and shapes and later layers will extract more complex image patterns. So for the output image we will take the **content** from later layers of CNN. For extracting the style of image, we take the correlations between different layers using [Gram Matrix](https://en.wikipedia.org/wiki/Gramian_matrix)
  
 Initially, we take any random image as target(or taking the content image would be useful) and compute the **Content loss** and **Style loss** and decreasing these losses we would reach the perfect target image that has the style of one image and content of other image.


  ## ImageProcessingExtensions.py

    ### **Purpose**
    Demonstates image processing techniques, tools, and results with OpenCV and brought to life as a GUI by PyQt5. Each button on the GUI has a unique and special purpose. 

    ## **Usage**
    You can click through the buttons and see the difference in the images. For example, sobel X and sobel Y are used to calculate the magnitude of the image for edge detection. 


 
