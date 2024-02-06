# Phone-Locator
A vision model to locate phones on a RGB image.

The model locates the phones by computing the center of the phone's position on the image, on normalized coordinates. The train_phone_finder.py expects a folder containing training images and a labels.txt file with the labels of each image on the format (file_name (x coordinate) (y coordinate)). The find_phone.py script expects a path to a single image as input and prints the predicted location for this image on the terminal.

## Approaches

The initial approach taken was to build a highly customizable convolutional network, where its hyper-parameters where tuned using [Optuna](https://optuna.readthedocs.io/en/stable/). This initial approach didn't got the desired performance, and it was rapidly discarded.

The second approach was to build another convolutional network with a ResNet-like architecture. This architecture, simpler and with less hyper-parameters to tune, was optimized, again, employing algorithms from Optuna. This approach produced a better performance than the first one, but still it was not satisfactory.

The third and final approach was to use a pre-trained model. The chosen model was AlexNet due to its high performance with a smaller size, balancing both performance and speed. The final layer was replaced with a custom linear layer with 2 outputs, then this layer was trained on the training images, and finally the model was fine-tuned on the same dataset.

For training, due to the small size of the dataset used, the train set was augmented through custom transformations, which also transform the labels to keep the correct positions. Additionally, because of the high risk of overfitting, the training process was regularized by employing early stopping and learning rate scheduling.

## Approaches not taken

An idea to produce a model with high performance and smaller size was to implement distilation model from a big model to a smaller one. However, there was no enough time to implement this idea.

Another idea, not taken due to the time it takes to implement, is to build a highly customizable convolutional block, and build models with the said blocks. Hence, we begin by building a model composed of only one of the mentioned blocks, optimizing the model to determine the best architecture with the given constraints, then add to the optimized model another block and optimize it and so on, finally obtaining a lego-like optimized model.

A final idea was to use model explainability techniques to determine the most relevant features on the model, then building a model that tries simplifies the original one by only implementing the relevant features of the previous model, then adding some more layers to improve perfomance, then repeat the process iteratively until there're no more relevant features to extract.

## Next steps

One inmediate next step may be to implement some of the ideas mentioned before. Implementing them require time and good analysis skills, but they could lead to excelent results.

Another way to improve the models is to collect more data. The small amount of data used makes very hard to produce satisfactory results, even when using powerful pre-trained models, so it seems like a natural next step.

This data may be collected by adding the images provided by the users to the training set, and emploting some human resources to label it. Alternatively, we can generate images with similar characteristics by employing generative AI, then using a object detection model to build the labels for the images, and making humans to check that the labels are correct, and to select the high-quality images for the train set.

When collecting data from the customer, we can make the images pass through an API that will feed the image to the model, while also storing the data on an image database. To employ this approach in an ethical way, however, we must at least make clear to the customer how their data will be store and te purpose for doing so.

## Dependencies

The scripts employ several functions and classes to work. All the essential python code is stored in the modules folder. For transparency, the code employed on experimentation is saved in the experiments folder. However, some experiments were run on the cloud and then adapted to work using the scripts on the modules folder.

Additionally, the project uses Python 3.11.5 and the following packages:
- [pytorch](https://pytorch.org/) == 2.1.0
- [torchvision](https://pytorch.org/vision/stable/index.html) == 0.15.2
- [torchinfo](https://github.com/TylerYep/torchinfo) == 1.8.0 (used only for experimentation)
- [opencv](https://opencv.org/) == 4.6.0
- [pandas](https://pandas.pydata.org/docs/index.html) == 1.5.3
- [numpy](https://numpy.org/) == 1.23.5
- [matplotlib](https://matplotlib.org/) == 3.6.2 (used only for experimentation)
- [optuna](https://optuna.readthedocs.io/en/stable/) == 3.4.0 (used only for experimentation)
- Also employes the built-in libraries [os](https://docs.python.org/3.10/library/os.html), [sys](https://docs.python.org/3/library/sys.html) and [typing](https://docs.python.org/3/library/typing.html)