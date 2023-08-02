# 404 CTF Hackathon: AI-Powered Code Decryption

## Introduction
In May 2023, I took part in the 404 CTF Hackathon where I endeavored to decrypt a code using the prowess of artificial intelligence. Given a folder of handwritten digit images, my task was to create an AI model adept at detecting and classifying these digits. The project showcases the might of AI in both code decryption and pattern recognition.

---

## Table of Contents
1. [Data Collection](#data-collection)
2. [Data Preprocessing](#data-preprocessing)
3. [AI Model Development](#ai-model-development)
4. [Model Training](#model-training)
5. [Model Validation and Fine-tuning](#model-validation-and-fine-tuning)
6. [Testing the Model on Challenge Images](#testing-the-model-on-challenge-images)
7. [Code Decryption](#code-decryption)

---

## Data Collection

To kickstart this project, I embarked on collecting a rich dataset of handwritten digit images, paying attention to encompass various writing styles and unique digit representations. The MNIST database, a renowned benchmark for handwritten digit recognition, enriched my dataset. I also contemplated creating a bespoke dataset, accumulating handwritten digit images from diverse sources.

---

## Data Preprocessing

With my dataset assembled, I transitioned to preprocessing the images to prepare them for AI model training. This phase encapsulated converting images to grayscale, resizing them to a consistent 28x28 pixel dimension, and normalizing pixel intensities.

---

## AI Model Development

Tackling the task of digit recognition necessitated employing Convolutional Neural Networks (CNNs), renowned for their prowess in image processing. The model's architecture comprised convolutional, pooling, and fully connected layers, tailored for recognizing the intricacies of handwritten digits.

---

## Model Training

Post architectural setup, the training phase ensued where the model learned from the dataset. Iterative learning was underpinned by backpropagation and algorithms like stochastic gradient descent, honing the model's predictive capabilities.

---

## Model Validation and Fine-tuning

To certify the model's robustness, I performed rigorous validation and fine-tuning. By assessing the model against a reserved validation set and tweaking hyperparameters accordingly, I ensured the model's resilience against diverse data.

---

## Testing the Model on Challenge Images

The trained model was then unleashed on the challenge images from the 404 CTF. Feeding the images to the model yielded digit predictions, elucidating the hidden code in the images.

---

## Code Decryption

The AI model's precise predictions were instrumental in decrypting the concealed code in the challenge images. Analyzing the predicted digits, I unveiled the encrypted sequence, a testament to AI's capabilities in addressing intricate challenges.

---

**Closing Note**: This project epitomizes the confluence of hackathon challenges with state-of-the-art AI techniques. Not only did it spotlight the utility of AI in decoding challenges but also underscored my aptitude in crafting effective AI solutions for real-world predicaments.
