# SE 2024 Group Project: Image Style Generator
## Setup
  ### Environment and Packages
  Project Language: Python
  
  #### Packages: 
  Default versions installed from python pip should work. - 2024/4

  For the main project demo: 

  ```
  python3 -m pip install numpy
  ```
  ```
  python3 -m pip install PyQt5
  ```
  ```
  python3 -m pip install torch
  ```
  ```
  python3 -m pip install torchvision
  ```
  If you want to try ImageProcessingExtensions.py you will also have to install **OpenCV**. 
  
  After the environment is set up, you can run the `main.py` file to see a quick demo of the project. In the search bar you can type "Earth" and view the different styles in our demo. 

  Included in the StylizeImg.py file is the source code for image styline with a VGG pretrained model. 

  ## Final Notes: 
  The original version of the code allows the user to search for images in NASA's image database and allows you to change the styles of the images. 

  TODO: I will probably have to modify the code to better fit the project topic. 

  Current Idea for the Final Version (Tentative): 
  - Allow user to input a content image. 
  - Allow user to use search engine to search for a style image. (Artwork or textures)
  - Generate and display an image that merges the content of the content image and the style of the style image. 
  