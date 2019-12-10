import os, json, argparse, numpy
import tensorflow as tf

from models import create_model
from data import load_data

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
import keras.backend

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--exp", default = "default")
  parser.add_argument("--gpu", default = "0")
  parser.add_argument("--epochs", default = 16, type = int)
  parser.add_argument("--batch", default = 4, type = int)
  parser.add_argument("--optimizer", default = "sgdm")
  parser.add_argument("--model", default = "vgg16")
  parser.add_argument("--siamese", default = "share")
  parser.add_argument("--weights", default = "imagenet")
  parser.add_argument("--module", default = "subtract")
  parser.add_argument("--activation", default = "tanh")
  parser.add_argument("--regularizer", default = "l2")

  FLAGS = parser.parse_args()
  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

  os.system("rm -rf ../exp-facematch/" + FLAGS.exp + "/")
  os.system("mkdir -p ../exp-facematch/" + FLAGS.exp + "/")
  json.dump(FLAGS.__dict__, open("../exp-facematch/" + FLAGS.exp + "/arguments.json", "w"))

  print("Loading data...")
  X, Y = load_data(FLAGS)
  print("Creating model...")
  model = create_model(FLAGS)
  model.summary()
  
  # Creates a graph.
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
  c = tf.matmul(a, b)
  # Creates a session with log_device_placement set to True.
  sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
  # Runs the op.
  print(sess.run(c))

  #shapes_count = int(numpy.sum([numpy.prod(numpy.array([s if isinstance(s, int) else 1 for s in l.output_shape])) for l in model.layers]))

  #memory = shapes_count * 4

  #print(memory)

  if FLAGS.optimizer == "sgd":
    from keras.optimizers import SGD
    optimizer = SGD(lr = 0.001)
  elif FLAGS.optimizer == "sgdm":
    from keras.optimizers import SGD
    optimizer = SGD(lr = 0.001, momentum = 0.9)
  elif FLAGS.optimizer == "adam":
    from keras.optimizers import Adam
    optimizer = Adam(lr = 0.01)
  else:
    raise NotImplementedError
  
  #keras.backend.get_session().run(tf.global_variables_initializer())
  #init=tf.global_variables_initializer()
  tf.GraphKeys.VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES

  callbacks = [
    ModelCheckpoint("../exp-facematch/" + FLAGS.exp + "/weights.hdf5", monitor = "val_acc", save_best_only = False, save_weights_only = True),
    TensorBoard(log_dir = "../exp-facematch/" + FLAGS.exp + "/logs/"),
    ReduceLROnPlateau(monitor = "val_acc", factor = 0.5, patience = 2)
  ]

  #run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)

  print("Compile...")
  model.compile(optimizer, loss = "binary_crossentropy", metrics = ["accuracy"]) #, options = run_opts
  
  print("Train...")
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    model.fit(X["train"], Y["train"], validation_data = [X["valid"], Y["valid"]], epochs = FLAGS.epochs, batch_size = FLAGS.batch, callbacks = callbacks, verbose = 1)
  
  print("Finished")
