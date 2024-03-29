"""Functions for downloading and reading MNIST data."""
import gzip
import os
import urllib.request
import numpy

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'


def maybe_download(filename, work_directory):
  """Download the data from Yann's website, unless it's already here."""
  if not os.path.exists(work_directory):
    os.mkdir(work_directory)
  filepath = os.path.join(work_directory, filename)
  if not os.path.exists(filepath):
    filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
    statinfo = os.stat(filepath)
    print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
  return filepath


def _read32(bytestream):
  dt = numpy.dtype(numpy.uint32).newbyteorder('>')
  return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(filename):
  """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    magic = _read32(bytestream)
    if magic != 2051:
      raise ValueError(
          'Invalid magic number %d in MNIST image file: %s' %
          (magic, filename))
    num_images = _read32(bytestream)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    buf = bytestream.read(rows * cols * num_images)
    data = numpy.frombuffer(buf, dtype=numpy.uint8)
    data = data.reshape(num_images, rows, cols, 1)
    return data


def dense_to_one_hot(labels_dense, num_classes=10):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot


def extract_labels(filename, one_hot=False):
  """Extract the labels into a 1D uint8 numpy array [index]."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    magic = _read32(bytestream)
    if magic != 2049:
      raise ValueError(
          'Invalid magic number %d in MNIST label file: %s' %
          (magic, filename))
    num_items = _read32(bytestream)
    buf = bytestream.read(num_items)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8)
    if one_hot:
      return dense_to_one_hot(labels)
    return labels


def norm_minimax(m):
    mi = numpy.min(m)
    ma = numpy.max(m)
    return (m - mi) * (1 / (ma-mi))


def make_overlap_diff_texture(images):
    if images.shape[0] % 2 == 1:
        images = images[:images.shape[0] - 1]
    half = images.shape[0] // 2
#    freq1 = numpy.zeros((images.shape[1], images.shape[2]))
#    freq2 = numpy.zeros((images.shape[1], images.shape[2]))
#    wave_pos1 = images.shape[1] * 2 // 3
#    wave_pos2 = images.shape[1] * 8 // 9
#    freq1[wave_pos1][wave_pos1] = 1
#    freq2[wave_pos2][-wave_pos2] = 1
#    t1 = norm_minimax(numpy.real(numpy.fft.ifft2(freq1))) + 0.01
#    t2 = norm_minimax(numpy.real(numpy.fft.ifft2(freq2))) + 0.01
#    img1 = norm_minimax(images[:half] * numpy.expand_dims(numpy.repeat([t1], half, axis=0), axis=3))
#    img2 = numpy.expand_dims(numpy.repeat([t2], half, axis=0), axis=3)
#     return norm_minimax(numpy.where(img1 < 0.01, img2, img1)) * 255.
    left_shift = numpy.zeros_like(images[:half])
    left_shift[:, :, 4:] = images[:half, :, :-4]
    right_shift = numpy.zeros_like(images[half:])
    right_shift[:, :, :-4] = images[half:, :, 4:]
    overlap = norm_minimax(left_shift) + norm_minimax(right_shift)
    overlap[overlap > 1] = 1
    return overlap * 255




def make_overlap_label(labels):
    half = labels.shape[0] // 2
    if labels.shape[0] % 2 == 1:
        labels = labels[:labels.shape[0] - 1]
    l = labels[:half] + labels[half:]
    l[l > 1] = 1
    return l


class DataSet(object):

  def __init__(self, images, labels, fake_data=False):
    if fake_data:
      self._num_examples = 10000
    else:
      assert images.shape[0] == labels.shape[0], (
          "images.shape: %s labels.shape: %s" % (images.shape,
                                                 labels.shape))
      self._num_examples = images.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      assert images.shape[3] == 1
      images = images.reshape(images.shape[0],
                              images.shape[1] * images.shape[2])
      # Convert from [0, 255] -> [0.0, 1.0].
      images = images.astype(numpy.float32)
      images = numpy.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1.0 for _ in range(784)]
      fake_label = 0
      return [fake_image for _ in range(batch_size)], [
          fake_label for _ in range(batch_size)]
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]

class SemiDataSet(object):
    def __init__(self, images, labels, n_labeled):
        self.n_labeled = n_labeled

        # Unlabled DataSet
        self.unlabeled_ds = DataSet(images, labels)

        # Labeled DataSet
        self.num_examples = self.unlabeled_ds.num_examples
        indices = numpy.arange(self.num_examples)
        shuffled_indices = numpy.random.permutation(indices)
        images = images[shuffled_indices]
        labels = labels[shuffled_indices]
        y = numpy.array([numpy.arange(10)[l == 1][0] for l in labels])
        idx = indices[y == 0][:5]
        n_classes = y.max() + 1
        n_from_each_class = int(n_labeled / n_classes)
        i_labeled = []
        for c in range(n_classes):
            i = indices[y == c][:n_from_each_class]
            i_labeled += list(i)
        l_images = images[i_labeled]
        l_labels = labels[i_labeled]
        self.labeled_ds = DataSet(l_images, l_labels)

    def next_batch(self, batch_size):
        unlabeled_images, _ = self.unlabeled_ds.next_batch(batch_size)
        if batch_size > self.n_labeled:
            labeled_images, labels = self.labeled_ds.next_batch(self.n_labeled)
        else:
            labeled_images, labels = self.labeled_ds.next_batch(batch_size)
        images = numpy.vstack([labeled_images, unlabeled_images])
        return images, labels


def read_data_sets(train_dir, n_labeled=100, fake_data=False, one_hot=False):
    class DataSets(object):
        pass
    data_sets = DataSets()

    if fake_data:
        data_sets.train = DataSet([], [], fake_data=True)
        data_sets.validation = DataSet([], [], fake_data=True)
        data_sets.test = DataSet([], [], fake_data=True)
        return data_sets

    TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
    TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
    TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
    TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
    VALIDATION_SIZE = 0

    local_file = maybe_download(TRAIN_IMAGES, train_dir)
    train_images = make_overlap_diff_texture(extract_images(local_file))

    local_file = maybe_download(TRAIN_LABELS, train_dir)
    train_labels = make_overlap_label(extract_labels(local_file, one_hot=one_hot))

    local_file = maybe_download(TEST_IMAGES, train_dir)
    test_images = make_overlap_diff_texture(extract_images(local_file))

    local_file = maybe_download(TEST_LABELS, train_dir)
    test_labels = make_overlap_label(extract_labels(local_file, one_hot=one_hot))

    validation_images = train_images[:VALIDATION_SIZE]
    validation_labels = train_labels[:VALIDATION_SIZE]
    train_images = train_images[VALIDATION_SIZE:]
    train_labels = train_labels[VALIDATION_SIZE:]

    data_sets.train = SemiDataSet(train_images, train_labels, n_labeled)
    data_sets.validation = DataSet(validation_images, validation_labels)
    data_sets.test = DataSet(test_images, test_labels)

    return data_sets
