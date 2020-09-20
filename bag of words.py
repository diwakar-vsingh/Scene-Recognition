import cv2
import numpy as np
import pickle
from utils import load_image, load_image_gray
import cyvlfeat as vlfeat
import sklearn.metrics.pairwise as sklearn_pairwise
from sklearn.svm import LinearSVC
from IPython.core.debugger import set_trace
from sklearn.neighbors import KNeighborsClassifier


def get_tiny_images(image_paths, patch_size):
  """
  This feature is inspired by the simple tiny images used as features in
  80 million tiny images: a large dataset for non-parametric object and
  scene recognition. A. Torralba, R. Fergus, W. T. Freeman. IEEE
  Transactions on Pattern Analysis and Machine Intelligence, vol.30(11),
  pp. 1958-1970, 2008. http://groups.csail.mit.edu/vision/TinyImages/

  To build a tiny image feature, simply resize the original image to a very
  small square resolution, e.g. 16x16. You can either resize the images to
  square while ignoring their aspect ratio or you can crop the center
  square portion out of each image.a Making the tiny images zero mean and
  unit length (normalizing them) will increase performance modestly.

  Useful functions:
  -   cv2.resize
  -   use load_image(path) to load a RGB images and load_image_gray(path) to
      load grayscale images

  Args:
  -   image_paths: list of N elements containing image paths

  Returns:
  -   feats: N x d numpy array of resized and then vectorized tiny images
            e.g. if the images are resized to 16x16, d would be 256
  """
  
  #############################################################################
  
  feats = []
  for paths in image_paths:
  # Resize the images to square (16x16) while ignoring the aspect ratio
    I = cv2.resize(load_image_gray(paths), (patch_size, patch_size)).flatten()
    
    # Making tiny images zero mean
    Im = I - np.mean(I)

    # Normalizing tiny images to have unit length
    In = Im/np.linalg.norm(Im)
    feats.append(In)

  feats = np.array(feats)
  
  return feats
  #############################################################################

def build_vocabulary(image_paths, vocab_size):
  """
  This function will sample SIFT descriptors from the training images,
  cluster them with kmeans, and then return the cluster centers.

  Useful functions:
  -   Use load_image(path) to load RGB images and load_image_gray(path) to load
          grayscale images
  -   frames, descriptors = vlfeat.sift.dsift(img)
        http://www.vlfeat.org/matlab/vl_dsift.html
          -  frames is a N x 2 matrix of locations, which can be thrown away
          here (but possibly used for extra credit in get_bags_of_sifts if
          you're making a "spatial pyramid").
          -  descriptors is a N x 128 matrix of SIFT features
        Note: there are step, bin size, and smoothing parameters you can
        manipulate for dsift(). We recommend debugging with the 'fast'
        parameter. This approximate version of SIFT is about 20 times faster to
        compute. Also, be sure not to use the default value of step size. It
        will be very slow and you'll see relatively little performance gain
        from extremely dense sampling. You are welcome to use your own SIFT
        feature code! It will probably be slower, though.
  -   cluster_centers = vl_kmeans(X, K)
          http://www.vlfeat.org/matlab/vl_kmeans.html
            -  X is a N x d numpy array of sampled SIFT features, where N is
               the number of features sampled. N should be pretty large!
            -  K is the number of clusters desired (vocab_size)
               cluster_centers is a K x d matrix of cluster centers. This is
               your vocabulary.

  Args:
  -   image_paths: list of image paths.
  -   vocab_size: size of vocabulary

  Returns:
  -   vocab: This is a vocab_size x d numpy array (vocabulary). Each row is a
      cluster center / visual word
  """
  # Load images from the training set. To save computation time, you don't
  # necessarily need to sample from all images, although it would be better
  # to do so. You can randomly sample the descriptors from each image to save
  # memory and speed up the clustering. Or you can simply call vl_dsift with
  # a large step size here, but a smaller step size in get_bags_of_sifts.
  #
  # For each loaded image, get some SIFT features. You don't have to get as
  # many SIFT features as you will in get_bags_of_sift, because you're only
  # trying to get a representative sample here.
  #
  # Once you have tens of thousands of SIFT features from many training
  # images, cluster them with kmeans. The resulting centroids are now your
  # visual word vocabulary.

  #############################################################################
  dim = 128      # length of the SIFT descriptors that you are going to compute.
  descriptors = []
  # Sampling all the images
  for i in range(len(image_paths)):
    desc = vlfeat.sift.dsift(load_image_gray(image_paths[i]), fast=True, step=10)[1]
    if i==0:
      descriptors = desc
    else:
      descriptors = np.vstack((descriptors, desc))
  descriptors = np.random.permutation(descriptors)
  vocab = vlfeat.kmeans.kmeans(descriptors.astype(np.float64),vocab_size)
  return vocab

  #############################################################################


def get_bags_of_sifts(image_paths, vocab_filename):
  """
  This feature representation is described in the handout, lecture
  materials, and Szeliski chapter 14.
  You will want to construct SIFT features here in the same way you
  did in build_vocabulary() (except for possibly changing the sampling
  rate) and then assign each local feature to its nearest cluster center
  and build a histogram indicating how many times each cluster was used.
  Don't forget to normalize the histogram, or else a larger image with more
  SIFT features will look very different from a smaller version of the same
  image.

  Useful functions:
  -   Use load_image(path) to load RGB images and load_image_gray(path) to load
          grayscale images
  -   frames, descriptors = vlfeat.sift.dsift(img)
          http://www.vlfeat.org/matlab/vl_dsift.html
        frames is a M x 2 matrix of locations, which can be thrown away here
          (but possibly used for extra credit in get_bags_of_sifts if you're
          making a "spatial pyramid").
        descriptors is a M x 128 matrix of SIFT features
          note: there are step, bin size, and smoothing parameters you can
          manipulate for dsift(). We recommend debugging with the 'fast'
          parameter. This approximate version of SIFT is about 20 times faster
          to compute. Also, be sure not to use the default value of step size.
          It will be very slow and you'll see relatively little performance
          gain from extremely dense sampling. You are welcome to use your own
          SIFT feature code! It will probably be slower, though.
  -   assignments = vlfeat.kmeans.kmeans_quantize(data, vocab)
          finds the cluster assigments for features in data
            -  data is a M x d matrix of image features
            -  vocab is the vocab_size x d matrix of cluster centers
            (vocabulary)
            -  assignments is a Mx1 array of assignments of feature vectors to
            nearest cluster centers, each element is an integer in
            [0, vocab_size)

  Args:
  -   image_paths: paths to N images
  -   vocab_filename: Path to the precomputed vocabulary.
          This function assumes that vocab_filename exists and contains an
          vocab_size x 128 ndarray 'vocab' where each row is a kmeans centroid
          or visual word. This ndarray is saved to disk rather than passed in
          as a parameter to avoid recomputing the vocabulary every run.

  Returns:
  -   image_feats: N x d matrix, where d is the dimensionality of the
          feature representation. In this case, d will equal the number of
          clusters or equivalently the number of entries in each image's
          histogram (vocab_size) below.
  """

  
  ############################################################################
  # load vocabulary
  with open(vocab_filename, 'rb') as f:
    vocab = pickle.load(f)

  feats = []
  for i in range(len(image_paths)):
    frames, descriptors = vlfeat.sift.dsift(load_image_gray(image_paths[i]), fast=True, step=5)
    quantize = vlfeat.kmeans.kmeans_quantize(descriptors.astype(np.float64), vocab)
    hist = np.histogram(quantize, bins=np.arange(len(vocab) + 1))[0]
    feats = np.append(feats, hist/np.max(hist))
  feats = feats.reshape((len(image_paths), vocab.shape[0]))
  
  return feats

  #############################################################################

def nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats, k_size, DM):
  """
  This function will predict the category for every test image by finding
  the training image with most similar features. Instead of 1 nearest
  neighbor, you can vote based on k nearest neighbors which will increase
  performance (although you need to pick a reasonable value for k).

  Useful functions:
  -   D = sklearn_pairwise.pairwise_distances(X, Y)
        computes the distance matrix D between all pairs of rows in X and Y.
          -  X is a N x d numpy array of d-dimensional features arranged along
          N rows
          -  Y is a M x d numpy array of d-dimensional features arranged along
          N rows
          -  D is a N x M numpy array where d(i, j) is the distance between row
          i of X and row j of Y

  Args:
  -   train_image_feats:  N x d numpy array, where d is the dimensionality of
          the feature representation
  -   train_labels: N element list, where each entry is a string indicating
          the ground truth category for each training image
  -   test_image_feats: M x d numpy array, where d is the dimensionality of the
          feature representation. You can assume N = M, unless you have changed
          the starter code
  -   metric: (optional) metric to be used for nearest neighbor.
          Can be used to select different distance functions. The default
          metric, 'euclidean' is fine for tiny images. 'chi2' tends to work`
          well for histograms

  Returns:
  -   test_labels: M element list, where each entry is a string indicating the
          predicted category for each testing image
  """

  #############################################################################
  
  # D = sklearn_pairwise.pairwise_distances(train_image_feats, test_image_feats)
  
  # index = np.argmin(D, axis=0)

  # test_labels = []
  # for i in range(len(test_image_feats)):
  #   test_labels.append(train_labels[index[i]])
  
  knn = KNeighborsClassifier(n_neighbors=k_size, p=DM)
  knn.fit(train_image_feats, train_labels)
  test_labels = knn.predict(test_image_feats)  
  
  return  test_labels

  #############################################################################


def svm_classify(train_image_feats, train_labels, test_image_feats):
  """
  This function will train a linear SVM for every category (i.e. one vs all)
  and then use the learned linear classifiers to predict the category of
  every test image. Every test feature will be evaluated with all 15 SVMs
  and the most confident SVM will "win". Confidence, or distance from the
  margin, is W*X + B where '*' is the inner product or dot product and W and
  B are the learned hyperplane parameters.

  Useful functions:
  -   sklearn LinearSVC
        http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
  -   svm.fit(X, y)
  -   set(l)

  Args:
  -   train_image_feats:  N x d numpy array, where d is the dimensionality of
          the feature representation
  -   train_labels: N element list, where each entry is a string indicating the
          ground truth category for each training image
  -   test_image_feats: M x d numpy array, where d is the dimensionality of the
          feature representation. You can assume N = M, unless you have changed
          the starter code
  Returns:
  -   test_labels: M element list, where each entry is a string indicating the
          predicted category for each testing image
  """

  #############################################################################
  
  # All the defined Categories
  categories = list(set(train_labels))

  # Construct 1 vs all linear SVMs classification for each category
  svms = {cat: LinearSVC(random_state=None, tol=1e-3, loss='squared_hinge', C=1) for cat in categories}
  
  for cat in categories:
    labels = []
    for c in train_labels:
      if (c == cat):
        labels.append(1)
      else:
        labels.append(-1)
    labels = np.array(labels)   # Classifying test data as 1 and -1 for each category
    svms[cat].fit(train_image_feats, labels)    # Fit the model according to the given training data for each category

  # Test image features are evaluated with all 15 SVMs
  confidences = np.array([svms[cat].decision_function(test_image_feats) for cat in categories])
  
  # Most confident SVM governs the label for the image
  labels_indices = np.argmax(confidences, axis=0)
  test_labels = [categories[index] for index in labels_indices]

  return test_labels

  #############################################################################
