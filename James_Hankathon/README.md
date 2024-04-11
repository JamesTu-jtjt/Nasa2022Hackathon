# James Hankathon Project: the JThankAi Solution
 Project page: https://2022.spaceappschallenge.org/challenges/2022-challenges/art-worlds/teams/james-hankathon/project
## Languages and Frameworks Used 
  * Python: coding language
  * NumPy: library for numerical calculations
  * Matplotlib: library for data visualization
  * Pytorch: a deep learning framework by Facebook AI Research Team for building neural networks
  * torchvision: package consists of popular datasets, model architectures, and common image transformations for computer vision
  * OpenCV: library for image processing
  * PyQt5: library for GUI programming and demonstration

## Setup
  After cloning, you can run the `main.py` file to see a quick demo of the project. In the search bar you can type "Earth" and view the different styles in our demo. Afterwards, you can run the `ImageProcessingExtensions.py` file to see a demo of basic image processing of imagery. 


# Code Review

  ## main.py
  ### **Usage**
    Go to the search bar on top and type **Earth** ; 
    Hit the **Go** button; 
    See the result of your search;
    Press the **Change style** button to see a wide range of stylistic changes to the Earth image. 

  ### **Logic Flow**
    Input keywords: Downloads the first image queried in "https://images.nasa.gov/" to the assets folder;  
    Pass the image and a given style image to the StylizeImg.py file to train; 
    Result is a new image of the two images, content and style, combined and visualized;  
    Utilizes PyQt5 to implement an interactive GUI for more accessible and observable results

  ## NasaAPICrawler.py

  ### **Purpose**
    the backbone of our search engine. uses API provided by NASA to crawl imagery from NASA's vast dataset.  


  ## StylizeImg.py

  ### **Purpose**
    StylizeImg.py is an image-combining algorithm that uses the VGG pre-trained AI model to implement the merging of 2 images. the resulting image is a blend of the content of one of the images and the style of the other, similar to a customized style filter based on the image. 

  ## ImageProcessingExtensions.py

  ### **Purpose**
    Demonstrates image processing techniques, tools, and results with OpenCV and brought to life as a GUI by PyQt5. Each button on the GUI has a unique and special purpose. 

  ### **Usage**
    You can click through the buttons and see the difference in the images. For example, Sobel X and Sobel Y are used to calculate the magnitude of the image for edge detection. 


 
