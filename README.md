# Deep-Learning
Lesson 3 Notes
In this lesson, we proceed by going over all the pieces that define convolutional layers, as well as discuss common practices in CNN architecture. We then close with a discussion on treating underfitting and overfitting models.


Contents
1	Convolutions
1.1	Filters
1.2	Filters in CNNs
1.3	Position Invariance
1.4	Zero-Padding
2	Convolutional Layer
2.1	Weight-Matrices
2.2	End-to-end
2.3	Feature Learning
2.4	Visualizing Convolutional Layers
3	Max-Pooling
4	Softmax
5	On Architecture
6	VGG16 Architecture
7	On Deeper Finetuning
8	Training a Better Model
8.1	Dropout
8.1.1	Aside: Pre-calculating Convolutional Layer Output
8.1.2	Updating Weights with Dropout
9	Overfitting
9.1	Data Augmentation
9.2	Batch Normalization
Convolutions
Mnist.png
Seven corner.png
Seven number.png
What exactly is a convolution?

Let's revisit Lesson 0 to solidify our understanding of what they are. Recall the MNIST dataset, a collection of 55,000 28x28 grey-scale images of handwritten digits with known labels. A greyscale image can simply be thought of as a matrix, with each element representing the corresponding pixel's location on the greyscale (the higher the whiter, the lower the darker).





Filters
Observe now the following 3x3 matrix,

Filter.png
which we can visualize as

Vis filter.png
Now imagine taking this 3x3 matrix and positioning it over a 3x3 area of an image, and let's multiply each overlapping value. Next, let's sum up these products, and let's replace the center pixel with this new value. If we slide this 3x3 matrix over the entire image, we can construct a new image by replacing each pixel in the same manner just described.

What does this new image look like? Consider the values in our 3x3 matrix. The top row is all -1's the middle row is all 1's and the bottom 0's. Then we would imagine that the brightest pixels in this new image are ones in a 3x3 area where the row above it is all 0 (black), and the center row is all 1 (white). This corresponds to top edges, and we can see that when we apply this matrix operation over the entire image, it does indeed highlight the top edges

This 3x3 matrix we apply to the image is what's called a filter. There are many kinds of filters that have been created over decades to detect different things. If we rotate our top edge filter, we can naturally detect different kinds of edges in different directions. Of course there are many kinds of different filters, and you can play around with some of them here.

These filters... Filters1.png Filters2.png

...result in these outputs: Filtered sevens.png

Let's now look at how to apply these ideas to neural networks.

Filters in CNNs
Now that we understand that filters can be used to identify particular visual "elements" of an image, it's easy to see why they're used in deep learning for image recognition. But how do we decide which kinds of filters are the most effective? Specifically, what filters are best at capturing the necessary detail from our image to classify it? We could proceed through trial and error utilizing many different filters ad infinitum and see which ones work best. But these filters are just matrices that we are applying to our input to achieve a desired output... therefore, given labelled input, we don't need to manually decide what filters work best at classifying our images, we can simply train a model to do so, using these filters as weights!

For example, we can start with 8 randomly generated filters; that is 8 3x3 matrices with random elements. Given labeled inputs, we can then use stochastic gradient descent to determine what the optimal values of these filters are, and therefore we allow the neural network to learn what things are most important to detect in classifying images.

Position Invariance
A particularly powerful characteristic of these convolutional filters is their positional invariance. Given that these filters act locally upon the input image, they will identify something like a "top-edge" anywhere in the image. This is important since it allows us to avoid making any assumption about the location or format of our images; with a well-trained CNN, we can locate a face whether it is in the center of the image or in a corner.

Zero-Padding
One thing we have neglected to mention is how these filters operate on the edge and corner pixels, given that the filter necessarily operates on the premise that there are 8 surrounding pixels. There is a variety of approaches to handling these edge/corner cases, but one of the most common and the one we'll use the most in this course is zero-padding. All zero-padding does is add a extra borders of zero pixels around the image prior to passing through a filter so that the output shape from the filter is the same as the input shape.

Convolutional Layer
We now know enough to understand what a convolutional layer is, and how it differs from a fully connected layer.

Weight-Matrices
Recall that a fully connected layer consisted of a matrix of weights that acted upon an input through matrix multiplication and produced an output vector that, subject to some bias, was then passed through a non-linearity of some sort (our activation layer).

In a fully connected layer, our weights were the elements of the matrix, and this matrix is used to transform the input vector into the output vector. Through training, the neural network learns what weights produce the outputs most in line with our expectations.

In a convolutional layer, our weights are still elements of matrices, but they are no longer performing transformations via matrix multiplication on the entire input. Rather, our weights belong to a set of matrices called filters that act on the input by performing the "overlapping" local element-wise multiplication operation we described above. When this filter is applied to the entire image, it has essentially created a new representation of the original image.

The output then of a convolutional layer is simply k representations of the original image, where k is the number of filters, and each k-th "representation" is simply the original image acted upon by the k-th filter.

End-to-end
As an example, let's walk through a forward-pass through a convolutional layer with 12 filters for a 28x28 grey-scale image.

Let us assume that each filter is 3x3, and we apply each filter to every applicable pixel in the image. In order to return an image of the same size as our input, we'd like to zero-pad our grey-scale image to produce a 30x30 image. This is just the same image bordered by zero pixels.

We now can pass our 30x30 image into the convolutional layer. At this point, each filter will pass through the image and produce a new image. Since there are 12 filters, there will be 12 new images, and our output is now a 28x28x12 'tensor' (which simply refers to a matrix with more than 2 dimensions); the final dimension represents the affect of each filter on the original image.

This new 28x28x12 "image" can now be passed again through a new convolutional layer to find even more complex structures in the image.

More often than not, our input images are in color, and so the input images may actually be something with dimension 224x224x3, in which case our filters actually have three layers themselves (in which case we would call them a '3d tensor', rather than a 'matrix'). There are a number of different factors and choices that go into understanding the exact algebra behind convolutional layers, as well as what dimension their outputs are, and we highly recommend exploring it further here.

Feature Learning
One way to conceptualize the output of a convolutional layer is in terms of features. A key advantage of neural networks as opposed to traditional machine learning techniques is that we can avoid feature engineering our raw input, i.e. constructing pre-determined features from our raw data that we think is important. For example, earlier we spoke of pre-designed filters that are meant to identify things like top-edges. When we apply these filters to images we can think of it as creating features from raw input, the features being "top-edges".

If we were to pick 12 of these pre-designed filters and apply them to an image, then pass those features to a machine learning algorithm, we've drastically diminished the potential predictive capability of our algorithm by determining a priori what features are most important, and discarding all other raw data.

In training convolutional neural networks, we allow stochastic gradient descent to find optimal values for filter weights based off of the raw data itself, and in this way we've allowed our network to "learn" what the best features are from the raw data given labelled output. This is incredibly powerful, and it removes the onerous task of manually designing features.

When we stack multiple convolutional layers on top of each other, we can think about this task being iterative; specifically, given the features created from layer one, what new "output" features are optimal to create from these "input" features? This is how neural networks learn "concepts". For example, we can think of the first layer as identifying things like edges, gradients, etc. Now, given features like edges and gradients as input, the second layer might then identify things like corners and contours. This proceeds until we've reached layers with filters that can now identify complex concepts like fur or text.

Visualizing Convolutional Layers
One problem with allowing neural networks to create their own optimal filters is that we often have no idea what it is these filters are actually detecting, given that they don't conform to recognizable pre-built filters (yet ironically, the fact that neural networks create unrecognizable filters that we would not manually conceive is what makes them so powerful).

Fortunately, we have some insight from Matthew Zeiler's Visualizing and Understanding Convolutional Networks and related works.

In Zeiler's paper, we are able to visualize the sorts of image features each filter in a Convolutional layer detects, such as edges, gradients, corners, etc. The Deep Visualization Toolbox is a great tool that allows us to play around with different images, and to see what image features activate certain filters. We can even see what imagenet images activate certain filters. We recommend playing around with this toolbox, and note the increasing complexity of the filters in deeper layers. When experimenting, remember that each of these filters has been learned by the neural network as important. This is incredibly powerful.


Max-Pooling
One operation common in CNN's is max pooling. Put simply, a max pooling layer reduces the dimensionality of images (resolution) by reducing the number of pixels in the image. It does so by replacing an entire NxN area of the input image with the maximum pixel value in that area. For example, given a 4x4 image, max-pooling over 2x2 subsections of the image would output a 2x2 image, where each pixel is the largest pixel in each of the 4 2x2 areas in the original image.

One reason we would like to do this is to "force" our neural network to look at larger areas of the image at a time. For example, instead of identifying fur and noses, we'd like for the network to start identifying cats and dogs. In order to make up for the information lost upon pooling, we'll typically increase the number of filters in subsequent convolutional layers.

The other reason we utilize pooling is simply to reduce the amount of parameters and computation load. It is also helpful in controlling overfitting.

As an example we can see max-pooled versions of our filtered sevens from above:

Maxpooling.png

Softmax
Last lesson, we learned that a neural network is simply a sequence of linear layers (such as fully connected or convolutional layers) that are connected by activation layers, which operate on intermediate outputs by passing them through non-linear functions. We've already talked about the ReLu (rectified linear unit) function, and this is often used in intermediate layers. This week we'd like to discuss softmax, which is typically used as our final activation layer as output.

The softmax function is defined as:

   exp(x)/sum(exp(x))
where x is an array of activations.

We require that these fully connected outputs be interpreted as a probability. Therefore, we transform each output as a portion of a total sum. However, instead of simply doing the standard proportion, we apply this non-linear exponential function for a very specific reason; namely, we would like to make our highest outputs as close to 1 as possible and our lower outputs as close to zero. You can imagine the softmax function as pushing the true linear proportions closer to either 1 or 0. Why would we want to do this? Recall that our label outputs we provided for training were one-hot encoded. Naturally, we would like our neural network outputs to mimic these outputs as much as possible, and the softmax function does this by pushing the largest proportion to 1 and the rest to zero.

Another reason why we wouldn't use the standard proportion is because the outputs of the final connected layer can be negative. While you could take the absolute value to construct probabilities, you would lose information in regards to comparing the raw values, while the exponential intrinsically takes care of this.

Perhaps the best way to get an understanding of this function's behavior is to play around with it in a spreadsheet - an example has been provided for you in entropy_example.xlsx.

On Architecture
We know now from the Universal Approximation Theorem that any large enough neural network can approximate any arbitrarily complex function. We also know that there exists methods such as stochastic gradient descent to calculate estimations for the parameters of these neural networks. Then it would appear that given any architecture, we should be able to solve any problem. So why then are we interested in learning different architectures?

While it's true that any architecture can solve any problem given enough time, some of them can learn to solve these problems much faster than others (and will likely generalize better) by having far less parameters. This is why we care about understanding architectures such as convolutional neural networks as opposed to trying to solve every problem with deep fully connected neural networks.

VGG16 Architecture
We can use Keras to give a summary of it's built in Vgg16 model

from keras.applications.vgg16 import VGG16
vgg = VGG16()
vgg.summary()
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
input_1 (InputLayer)             (None, 3, 224, 224)   0                                            
____________________________________________________________________________________________________
block1_conv1 (Convolution2D)     (None, 64, 224, 224)  1792        input_1[0][0]                    
____________________________________________________________________________________________________
block1_conv2 (Convolution2D)     (None, 64, 224, 224)  36928       block1_conv1[0][0]               
____________________________________________________________________________________________________
block1_pool (MaxPooling2D)       (None, 64, 112, 112)  0           block1_conv2[0][0]               
____________________________________________________________________________________________________
block2_conv1 (Convolution2D)     (None, 128, 112, 112) 73856       block1_pool[0][0]                
____________________________________________________________________________________________________
block2_conv2 (Convolution2D)     (None, 128, 112, 112) 147584      block2_conv1[0][0]               
____________________________________________________________________________________________________
block2_pool (MaxPooling2D)       (None, 128, 56, 56)   0           block2_conv2[0][0]               
____________________________________________________________________________________________________
block3_conv1 (Convolution2D)     (None, 256, 56, 56)   295168      block2_pool[0][0]                
____________________________________________________________________________________________________
block3_conv2 (Convolution2D)     (None, 256, 56, 56)   590080      block3_conv1[0][0]               
____________________________________________________________________________________________________
block3_conv3 (Convolution2D)     (None, 256, 56, 56)   590080      block3_conv2[0][0]               
____________________________________________________________________________________________________
block3_pool (MaxPooling2D)       (None, 256, 28, 28)   0           block3_conv3[0][0]               
____________________________________________________________________________________________________
block4_conv1 (Convolution2D)     (None, 512, 28, 28)   1180160     block3_pool[0][0]                
____________________________________________________________________________________________________
block4_conv2 (Convolution2D)     (None, 512, 28, 28)   2359808     block4_conv1[0][0]               
____________________________________________________________________________________________________
block4_conv3 (Convolution2D)     (None, 512, 28, 28)   2359808     block4_conv2[0][0]               
____________________________________________________________________________________________________
block4_pool (MaxPooling2D)       (None, 512, 14, 14)   0           block4_conv3[0][0]               
____________________________________________________________________________________________________
block5_conv1 (Convolution2D)     (None, 512, 14, 14)   2359808     block4_pool[0][0]                
____________________________________________________________________________________________________
block5_conv2 (Convolution2D)     (None, 512, 14, 14)   2359808     block5_conv1[0][0]               
____________________________________________________________________________________________________
block5_conv3 (Convolution2D)     (None, 512, 14, 14)   2359808     block5_conv2[0][0]               
____________________________________________________________________________________________________
block5_pool (MaxPooling2D)       (None, 512, 7, 7)     0           block5_conv3[0][0]               
____________________________________________________________________________________________________
flatten (Flatten)                (None, 25088)         0           block5_pool[0][0]                
____________________________________________________________________________________________________
fc1 (Dense)                      (None, 4096)          102764544   flatten[0][0]                    
____________________________________________________________________________________________________
fc2 (Dense)                      (None, 4096)          16781312    fc1[0][0]                        
____________________________________________________________________________________________________
predictions (Dense)              (None, 1000)          4097000     fc2[0][0]                        
====================================================================================================
Total params: 138,357,544
Trainable params: 138,357,544
Non-trainable params: 0
On Deeper Finetuning
Last week, we talked about finetuning the Vgg16 model built on Imagenet to classify Cats and Dogs, which involved removing the last fully connected layer and replacing it with one that gave two outputs. We then retrained that last layer to find optimal parameters.

Our philosophy behind doing this was that Vgg16 had learned at a high-level from imagenet how to identify things relevant to making classifications for Cats and Dogs, and we only replaced the last fully connected layer because we wanted to take this knowledge and apply it to this new classification task.

In fine-tuning for a classification task such as Statefarm, we may have to go deeper. The Statefarm dataset tasks you with identifying different activities a distracted driver may be involved in. This is not similar to the original imagenet challenge, and as such it's probably a smart idea to retrain even more fully connected layers. The idea here is that imagenet has learned to identify things that are not useful for classifying driver actions, and we would like to train them further to find things that are useful.

We typically don't touch the convolutional layers, as we find that the filters typically still work well for almost any classification task that uses standard photos. Vision tasks using line art, medical imaging, or other domains very different to standard photos are likely to require retraining convolutional layers, however.

Training a Better Model
We've now developed the basic knowledge needed to understand CNN's. Let's now look at some of the common issues in building models, and some approaches to increasing our model's performance.

We'll first define two important concepts:

Underfitting: This describes a model that lacks the complexity to accurately capture the complexity inherent in the problem you're trying to solve. We can recognize this when error on both the training and validation sets is too high.
Overfitting: This describes a model that is using too many parameters and has been trained too long. Specifically, it has learned how to match your exact training images to classes, but has become so specific that it is unable to generalize to similar images. This is easily recognizable when your training set accuracy is much higher than your validation.
Dropout
You'll notice in observing our metrics while training a Vgg model for Cats and Dogs that our training accuracy is usually lower than our validation. This points to underfitting, and the source of this in Vgg is relatively simple.

If you observe the Vgg layers in Keras, you'll notice a layer called Dropout. Dropout (which only happens during training) occurs after the activation layers, and randomly sets activations to zero. Why would we want to randomly delete parts of neural network? It turns out that this allows us to prevent overfitting. We can't overfit exactly to our training data when we're consistently throwing away information learned along the way. This allows our neural network to learn to generalize.

In the case of Vgg, the Dropout rate is set to 0.5. This seems quite large, and given the complexity of classification for Imagenet, it seems reasonable to want to make our dropout rate this high. However, for something much simpler like Cats and Dogs, we can see that we're underfitting, and we might have more success by retaining more information and lowering our dropout rate, and retraining.


Aside: Pre-calculating Convolutional Layer Output
We mentioned this last lesson, but when we are finetuning and modifying elements in our fully-connected layers, you'll find that you can save alot of time by pre-calculating the outputs from the convolutional layers. In keras, when you fit on the entire model, each forward pass necessarily goes through the convolutional layers, even if we're not updating them in the back-propagation. Therefore, when we're only training the fully connected layers, we can compute the conv layer outputs and save a lot of time by treating that as our training set for a network consisting of only our fully connected layers. Once we're satisfied with our results, we can simply load those weights back onto the original full network. (Remember, utils.py provides `save_array()` and `load_array()` to make it easy to quickly save and load arrays using bcolz.

A general rule of thumb to keep in mind is that most of our computation time lies in the convolutional layers, while our memory overhead lies in our dense layers.

Updating Weights with Dropout
Dropout is a technique of regularization. Like any other such method it trades some ability of your model to fit the training data in hopes that what it learns might generalize better to the data it has not seen. Beyond that, dropout also has some very nice unique properties that you can read more about here (section 2 is very interesting and can give you great intuition on why it works!)

Classical dropout is achieved by randomly disregarding some subset of nodes in a layer during training. The nodes which don't participate in producing a prediction and subsequently do not take part in calculating the gradient are selected at random and will vary from training on one example to another. During test time we want to utilize the full predictive capacity of our network and thus all nodes are active. Essentially what we are doing is we are averaging over the contributions of all nodes. If during training we set nodes to be inactive with probability p = 0.5, we now will have to make the weights twice as small as during training. The intuition behind this is very simple - if during train time we were teaching our network to predict '1' in the subsequent layer utilizing only 50% of its weights, now that it has all the weights at its disposal the contribution of each weight needs to only be half as big!

This was the classical dropout. What keras does is slightly different - it has something that can be referred to as inverted dropout. The weights are rescaled during training so that no rescaling needs to take place during test! This also has this nice property that you can move weights around calling get_weights and set_weights on a layer with easy and without any manipulations to the scale of the weights.

Thus, to summarize, regardless if you apply dropout to a layer, in keras the weights will always be of correct scale. This is not something that is evident from the lesson video - Jeremy assumed that keras would apply dropout in a classical way. Everything in the lesson still applies and the rescaling of weights is still 100% accurate should we be applying classical dropout but through the inner workings of keras this step can be disregarded (if you do the rescaling you will end up with weights that are either too small or to large!)

Overfitting
In general, once we've arrived at a model that is overfitting, there are five steps we should take to reduce it:

Add more data
Use data augmentation
Use architectures that generalize well
Add regularization
Reduce architecture complexity.
We'll talk about some of these steps in this lesson.

Data Augmentation
Earlier we mentioned that overfitting is a result of our network having learned too much of the specifics of our training set. In other words, we've created a model that relies too much on the specific qualities in our training set, and is unable to generalize and make predictions for similar images.

Data aug.png
Data augmentation is a method of addressing this. Simply put, data augmentation just alters each batch of our images. It does this through flipping, slightly changing hues, stretching, shearing, rotation, etc. and it does this in ways that make sense. By that we mean, it wouldn't make sense to vertically flip an image of a dog for generalization purposes, because people rarely if ever take upside down images of dogs. The same can be said for distortions; you don't want to apply so much as to make an image that is far beyond any reasonable image one would take of a cat or dog.


Keras allows you to implement data augmentation fairly easily by creating a data-augmentation batch generator. When constructing this generator, you have many choices in picking distortion parameters. Unfortunately, there is no quick and easy way to identify the optimal parameters for data augmentation: your best bet is to simply experiment.

In general, data augmentation is always a good idea to reduce overfitting.

Batch Normalization
Another good standard approach to reducing overfitting is Batch Normalization.

In general, the inputs to your neural network should always be normalized. Normalization is a process where given a set of data, you subtract from each element the mean value for that data set and divide it by the data set's standard deviation. By doing so, we put the input values onto the same "scale", in that all the values have been normalized into units of "# of standard deviations from the mean".

The reason we would want to put our inputs onto the same scale is because unbalanced inputs with a large range of magnitudes can typically cause instability in neural networks. An extremely large input can often cascade down through the layers. Typically such an imbalance creates gradients that are also wildly imbalanced, and this makes the optimization process difficult to prevent things like explosion. It also creates imbalanced weights. Normalizing inputs avoids this problem.

Often times with images, we don't worry about dividing by the standard deviation, but just subtract the mean.

Occasionally, these instabilities can arise during training. Imagine that at some point during training, we end up with one extremely large weight. This extremely large weight will then produce an extremely large output value for some element of the output vector, and this imbalance will again pass through the neural network and make it unstable.

One idea is to normalize the activation outputs. Unfortunately, this won't prevent SGD from trying to create an imbalanced weight again during the next back-propagation, and trying to solve the problem this way will just cause SGD to continuously try to undo this activation layer normalization. Batch normalize extends this idea with two additions:

after normalizing the activation layer, let's multiply the outputs by an arbitrarily set parameter, and add to that value an additional parameter, therefore setting a new standard deviation and mean
Make all four of these values (the mean and standard deviation for normalization, and the two new parameters to set arbitrary mean and standard deviations after the normalization) trainable.
This ensures that the weights don't tend to push very high or very low (since the normalization is included in the gradient calculations, so the updates are aware of the normalization). But it also ensures that if a layer does need to change the overall mean or standard deviation in order to match the output scale, it can do so.

By default, you should always include batch normalization, and all modern neural networks do so because:

Adding batchnorm to a model can result in 10x or more improvements in training speed
Because normalization greatly reduces the ability of a small number of outlying inputs to over-influence the training, it also tends to reduce overfitting.
