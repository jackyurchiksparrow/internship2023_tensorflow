# Image characters recognition

Instruction for making and using the project.

---

**Table of contents** 

1. [Data Analysis](#Data-Analysis)
2. [Action Plan](#action-plan)
3. [Choosing the right model, analyzing results and accuracy](#choosing-the-right-model-analyzing-results-and-accuracy)
    * [Step 1. Choose the initial point](#step-1-choose-the-initial-point)
    * [Step 2. Switching to Adam, reconsidering the batch size](#step-2-switching-to-adam-reconsidering-the-batch-size)
    * [Step 3. Editing the model](#step-3-editing-the-model)
4. [Usage Instruction](#usage-instruction)
5. [Running the model, analyzing results](#running-the-model-analyzing-results)
6. [User Information](#user-information)
7. [References](#references)

According to the [task](/internship2023.pdf), the necessity of recognizing 17 boxes per document is given. But the documents have already been scanned and the VIN-code boxes are already recognized. Hence, **the main objective** of the project is having a well-trained neural network (with saved model) and formalizing data.

## Data Analysis
---
To increase the accuracy of the future network the following data analysis was performed:
1. According to the International Organization for Standardization in ISO 3779 and ISO 4030 (which are also applicable for Ukraine) a vehicle VIN number [must only include](https://en.wikipedia.org/wiki/Vehicle_identification_number) numbers (0-9) and the letters (A-Z) except for (O, I, Q) to avoid their possible confusion with numerals (0, 1, 9) respectively. Reducing the amount of classes will also help in enhancing our model's performance.
2. Determining suitable dataset for training. The [EMNIST dataset](https://www.kaggle.com/datasets/crawford/emnist?select=emnist-byclass-mapping.txt) is a set of handwritten character digits derived from the NIST Special Database 19 and converted to a 28x28 pixel image which is ideal for our case. The only step that needs to be done is removing unnecessary symbols (small letters, capital O, I and Q). The amount of data is large enough to provide us with both training and testing sets.
3. Visualizing some of the EMNIST data gives us a clear understanding that **each** image there is **inverted** horizontally and **rotated** 90 anti-clockwise as well. Therefore, we need to take their **transpose** before teaching the model.
4. As the [task](/internship2023.pdf) demands we need the highest accuracy we can provide, so the best to use CNN (Convolutional neural network) to pay attention to as much image details possible. 
> Its built-in convolutional layer reduces the high dimensionality of images without losing its information; mainly used for applications in image and speech recognition.
5. The screenshot from the [task](/internship2023.pdf) states that final images will contain white character on a black background. There is also a white border. We **have to** take it into account and **either** consider _having our training data processed_ to the same 'format' or the _final data processed_ to the 'format' of training sets.

## Action Plan
---
1. Working with .csv data using [pandas](https://pandas.pydata.org/docs/), including parsing and getting rid of redundant data.
2. Using [emnist-balanced-train.csv](/emnist-balanced-train.csv) dataset given that we extract O, I, Q and small letters first. 
3. Transposing images using [numpy](https://numpy.org/doc/), as well as its other functional.
4. Choosing the right [model](#choosing-the-right-model-analyzing-results-and-accuracy).
5. Removing borders of images using truncation and [opencv](https://docs.opencv.org/4.x/); using [numpy](https://numpy.org/doc/) to invert colors to make training and final datasets look similar.
6. Visualizing results using [pandas](https://pandas.pydata.org/docs/) data frame, get ASCII index with ord() function and [pathlib](https://docs.python.org/3/library/pathlib.html).PurePosixPath to get POSIX path.

## Choosing the right model, analyzing results and accuracy
---
We're picking one of the standard CNN templates, providing the last layer with 36 neurons. [The original dataset](https://www.kaggle.com/datasets/crawford/emnist?select=emnist-byclass-mapping.txt) has 47 classes. We have got 36 as:
```
original_classes_size - three_capital_letters - small_letters = 36
```
In other words, our dataset must consist of:
```
10_digits + 26_letters = 36 characters
```
### Step 1. Choose the initial point
---
First, we have to look at the model behavior. Let us provide it with 20 _epochs_ (randomly, but only 20 to speed up the process of computation) and _batch_size_ of 32. However, we cannot neglect the batch size parameter value. High _batch_size_ will lead to poor generalization (but will also speed up the process). On the other hand, the smaller _batch_size_ will make us wait longer, but this way model can start learning before 'seeing' all of the data. Also, it's easier to track the moment of overfitting.

As we don't want to manually tune the _learning_rate_ the choice of optimizer will be on _Adaptive methods_.
In most cases _Adam_ works great as it combines advantages of _Adadelta_ and _RMSprop_. _RMSprop_ is quite fast (it's agile in adapting to changing the slope in gradient descent); we will try it first and consider _Adam_ (which is _Adadelta_ + momentum for smoothing) next. As a loss function we'll leave the usual _categorical_crossentropy_, which is a loss function for multi-class classification model where there are two or more output labels (exactly our case).
```
modelv1 = models.Sequential() 
modelv1.add(layers.Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1)))
modelv1.add(layers.MaxPooling2D((2,2)))
modelv1.add(layers.Conv2D(128, (3,3), activation='relu'))
modelv1.add(layers.MaxPooling2D((2,2)))
modelv1.add(layers.Conv2D(128, (3,3), activation='relu'))
modelv1.add(layers.Flatten())
modelv1.add(layers.Dense(128, activation='relu'))
modelv1.add(layers.Dense(36, activation='softmax'))

modelv1.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = modelv1.fit(train_images, train_labels, epochs=20, batch_size=32, 
                    validation_data = (test_images, test_labels))
```
Output:
|Epoch | Loss | Accuracy | Val_loss | Val_accuracy |
| :----| :----| :--------| :--------| :------------|
|1     | 0.324 | 0.8899  | 0.6735   | 0.8554       |
| 2    |0.324 | 0.8899  | 0.6735   | 0.8554       |
| 3    |0.324 | 0.8899  | 0.6735   | 0.8554       |
| ...   |... | ...  | ...   | ...       |
| 19   |0.1264 | 0.9530  | 1.0687   | 0.8594       |
| 20   |0.1262 | 0.9532  | 1.0189   | 0.8642       |

The _val_loss_ is high and _val_accuracy_ too. Possibly underfitting. Let us visualize the data:
```
loss = historyv2.history['loss']
val_loss = historyv2.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training_loss')
plt.plot(epochs, val_loss, 'b', label='Validation_loss')
plt.title('Training and validation loss')
plt.legend() 

plt.show()
```
![model1_plot](/src/model1_plot.png)

The image illustrates that the _training loss_ and _validation loss_ are both relatively high. This may indicate that the model is **underfitting**.

### Step 2. Switching to Adam, reconsidering the batch size
---
As mentioned above, we're switching to _Adam_. Also, due to lack in accuracy, I'm also decreasing the _batch_size_ from **32** to **20**.
```
modelv1.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = modelv1.fit(train_images, train_labels, epochs=20, batch_size=20, 
                    validation_data = (test_images, test_labels))
```
Output:
|Epoch | Loss | Accuracy | Val_loss | Val_accuracy |
| :----| :----| :--------| :--------| :------------|
|1     | 1.5471 | 0.5428  | 0.6542   | 0.7804       |
| 2    |0.8085 | 0.7367  | 0.5169   | 0.8211       |
| 3    |0.6723 | 0.7766  | 0.4535   | 0.8324       |
| ...   |... | ...  | ...   | ...       |
| 19   |0.3292 | 0.8797  | 0.3164   | 0.8832       |
| 20   |0.3236 | 0.8812  | 0.3171   | 0.8805       |

The results got worse. It looks like underfitting. Let us visualize the data:

![model1_plot2](/src/model1_plot2.png)

The model doesn't have a tendency of improving. It's still **underfitting**.

### Step 3. Editing the model
---
Using another template I'm increasing the amount of layers, as well as quantity of neurons in them, hoping that thus it will detect the distinctive features better. I will manually specify the _learning_rate_ to make the model learn more strictly. Other parameters remain the same.
```
modelv2 = models.Sequential()
modelv2.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28,28,1), padding="same"))
modelv2.add(layers.MaxPooling2D((2, 2)))
modelv2.add(layers.Conv2D(64, (3, 3), activation='relu', padding="same"))
modelv2.add(layers.MaxPooling2D((2, 2)))
modelv2.add(layers.Conv2D(128, (3, 3), activation='relu', padding="same"))
modelv2.add(layers.MaxPooling2D((2, 2)))
modelv2.add(layers.Conv2D(128, (3, 3), activation='relu', padding="same"))
modelv2.add(layers.MaxPooling2D((2, 2)))
modelv2.add(layers.Flatten())
modelv2.add(layers.Dropout(0.5))
modelv2.add(layers.Dense(512, activation='relu'))
modelv2.add(layers.Dense(36, activation='softmax'))

modelv2.compile(loss='categorical_crossentropy',
                optimizer=optimizers.Adam(learning_rate=1e-4),
                metrics=['accuracy'])

historyv2 = modelv2.fit(train_images, train_labels, epochs=20, batch_size=20, 
                    validation_data = (test_images, test_labels))
```
Output:
|Epoch | Loss | Accuracy | Val_loss | Val_accuracy |
| :----| :----| :--------| :--------| :------------|
|1     | 1.3236 | 0.6030   | 0.4521   | 0.8482       |
| 2    |0.5997 | 0.8065  | 0.3667   | 0.8693       |
| 3    |0.4751 | 0.8443  | 0.2873   | 0.8987       |
| ...   |... | ...  | ...   | ...       |
| 19   |0.1754 | 0.9357  | 0.1770   | 0.9399      |
| 20   |0.1668 | 0.9387  | 0.1684   | 0.9386       |

Now it looks better. The tendency of _val_loss_ decreasing and _val_accuracy_ decreasing orient us that the model is actually learning this time. Visualizing:

![model2_plot](/src/model2_plot.png)

According to the picture above, we've reached the right point between **underfitting** and **overfitting**. I will not try to increase accuracy by making the model more complicated as I didn't have any success in it. With that being said, I consider I've reached the optimal point. Saving the model:
```
model.save(r'model_letters_numbers_vin.h5')
```

## Usage Instruction
---
1. To run the program manually use the command:
```
py inference.py -f test_data
```

2. To run in the container use:
```
docker build -t intern .
docker run intern
```

## Running the model, analyzing results
---
![results](/src/results.png)

The [model](#step-3-editing-the-model) performed greatly on the test data and on my custom numbers and letters that I drew in paint, however, with the final dataset results are **12/14**. The only two images it hasn't classified right are hard to name even for me:

![errors](/src/errors.png)

The first one could be either G or 9, ugly C.
The second is probably 5 or S.

To summarize, the model has accuracy of **0.9386** on testing data and **0.857** on the final data (which is not right to consider because there was little data).

## User Information
---
Yurii Dzbanovskyi
* Email: uradzb@ukr.net
* Telegram: +38 096 874 17 18

## References
---
1. https://www.kaggle.com/code/ryanholbrook/overfitting-and-underfitting
2. https://medium.com/@enduranceprog/machine-vision-digits-94eb258c6ff8
3. https://habr.com/ru/articles/466565/
4. https://www.baeldung.com/cs/training-validation-loss-deep-learning
5. https://www.kaggle.com/datasets/crawford/emnist?select=emnist-byclass-mapping.txt
6. https://data-flair.training/blogs/handwritten-character-recognition-neural-network/