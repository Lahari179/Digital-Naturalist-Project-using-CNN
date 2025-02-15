
### GitHub README Description:  

# Digital Naturalist Using CNN  

### Overview  
Digital Naturalist is a web application designed to assist field naturalists and nature enthusiasts in identifying different species of birds, flowers, and mammals. Using a Convolutional Neural Network (CNN) model, the app provides accurate species predictions from uploaded images, helping users explore and learn about the natural world around them.

### Features  
- Identifies birds, flowers, and mammals with high accuracy (~97%).
- Utilizes CNN for image classification using TensorFlow and Keras.
- Built using Flask for a seamless web interface.
- Aids in conservation efforts by collecting and sharing species information.
- Raises alerts for crop protection from wildlife.

### Objective  
The project aims to provide a handy and reliable tool for naturalists to capture, identify, and share species information effortlessly. By leveraging deep learning, the application ensures accurate identification while promoting wildlife conservation.

### Tech Stack  
- **Deep Learning Model**: Convolutional Neural Network (CNN)
- **Libraries**: TensorFlow, Keras, ImageDataGenerator
- **Framework**: Flask (Python-based micro web framework)
- **Frontend**: HTML templates for user interaction
- **Development Environment**: Jupyter Notebook, Spyder IDE

### Installation and Setup  
1. Clone the repository:  
```bash
git clone https://github.com/yourusername/digital-naturalist-cnn.git
```
2. Navigate to the project directory:  
```bash
cd digital-naturalist-cnn
```
3. Install the required dependencies:  
```bash
pip install -r requirements.txt
```
4. Run the application:  
```bash
python app.py
```

### Usage  
- Upload an image of a bird, flower, or mammal.
- The app will process the image using the trained CNN model.
- The predicted species along with relevant information will be displayed.

### Dataset  
The model was trained on a custom dataset with 1595 images across different species, divided into subclasses for better classification accuracy. 80% of the dataset was used for training, while 20% was reserved for testing.

### Results  
- Achieved a testing accuracy of approximately 97%.
- The model performs well under varying conditions like illumination and complex backgrounds.

### Applications  
- Assists field naturalists in identifying species during hikes, canoe trips, and other excursions.
- Aids conservation groups in monitoring and protecting wild spaces.
- Provides crop protection alerts by identifying wildlife intrusions.

### Merits  
- High accuracy in image classification.
- Robust performance under diverse environmental conditions.
- Facilitates easy identification and sharing of species information.

### Limitations  
- Longer training times for CNN models.
- Hardware constraints may affect model performance on larger datasets.

### Future Enhancements  
- Expand the dataset to include more species and subclasses.
- Implement real-time image processing and predictions.
- Optimize model performance for mobile devices.

### Contributing  
Contributions are welcome! Please fork the repository and create a pull request with your proposed changes.


### Acknowledgments  
This project was developed as part of the Smart Bridge-Remote Summer Internship Program under the guidance of Smart Bridge. Special thanks to all contributors and mentors.

---
