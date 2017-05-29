import tensorflow as tf
s = tf.Session()
checkpoint = "checkpoints/last_checkpoint/hyperspectral_resnet.model-519"
saver = tf.train.import_meta_graph(checkpoint + ".meta")
saver.restore(s, checkpoint)
op = s.graph.get_tensor_by_name("bn1/beta:0")
print(s.run(op))
