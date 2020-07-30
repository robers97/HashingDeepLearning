import numpy as np
import time
import math
import os
import glob
import tensorflow as tf
import tf2onnx

#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

from config import config
from itertools import islice
#from scipy.sparse import csr_matrix
from util import data_generator, data_generator_tst

## Training Params
def main():
    global config
    feature_dim = config.feature_dim
    n_classes = config.n_classes
    hidden_dim = config.hidden_dim
    n_train = config.n_train
    n_test = config.n_test
    n_epochs = config.n_epochs
    batch_size = config.batch_size
    lr = config.lr
    #
    if config.GPUs=='':
        num_threads = config.num_threads
    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = config.GPUs
    #
    train_files = glob.glob(config.data_path_train)
    test_files = glob.glob(config.data_path_test)
    #
    x_idxs = tf.placeholder(tf.int64, shape=[None,2])
    x_vals = tf.placeholder(tf.float32, shape=[None])
    x = tf.SparseTensor(x_idxs, x_vals, [batch_size,feature_dim])
    y = tf.placeholder(tf.float32, shape=[None,n_classes],name="y")
    #
    W1 = tf.Variable(tf.truncated_normal([feature_dim,hidden_dim], stddev=2.0/math.sqrt(feature_dim+hidden_dim)))
    b1 = tf.Variable(tf.truncated_normal([hidden_dim], stddev=2.0/math.sqrt(feature_dim+hidden_dim)))
    layer_1 = tf.nn.relu(tf.sparse_tensor_dense_matmul(x,W1)+b1)
    #
    W2 = tf.Variable(tf.truncated_normal([hidden_dim,n_classes], stddev=2.0/math.sqrt(hidden_dim+n_classes)))
    b2 = tf.Variable(tf.truncated_normal([n_classes], stddev=2.0/math.sqrt(n_classes+hidden_dim)))
    logits = tf.matmul(layer_1,W2)+b2
    #
    k=1
    if k==1:
        top_idxs = tf.argmax(logits, axis=1)
    else:
        top_idxs = tf.nn.top_k(logits, k=k, sorted=False)[1]
    #
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    #
    train_step = tf.train.AdamOptimizer(lr).minimize(loss)
    #
    if config.GPUs=='':
        Config = tf.ConfigProto(inter_op_parallelism_threads=num_threads, intra_op_parallelism_threads=num_threads)
    else:
        Config = tf.ConfigProto()
        Config.gpu_options.allow_growth = True
    #
    sess = tf.Session(config=Config)
    sess.run(tf.global_variables_initializer())
    #
    training_data_generator = data_generator(train_files, batch_size, n_classes)
    steps_per_epoch = n_train//batch_size
    n_steps = n_epochs*steps_per_epoch
    n_check = 50
    #
    begin_time = time.time()
    total_time = 0
    counter = 0
    #
    with open(config.log_file, 'a') as out:
        for i in range(n_steps):
            if i%n_check==0:
                total_time+=time.time()-begin_time
                print('Finished ',i,' steps. Time elapsed for last',n_check,'batches = ',time.time()-begin_time)
                n_steps_val = n_test//batch_size
                test_data_generator = data_generator_tst(test_files, batch_size)
                tmp_k = 0
                for h in range(20):  # a few test batches to check the precision 
                    idxs_batch, vals_batch, labels_batch = next(test_data_generator)
                    top_k_classes = sess.run(top_idxs, feed_dict={x_idxs:idxs_batch, x_vals:vals_batch})
                    tmp_k += np.mean([len(np.intersect1d(top_k_classes[j],labels_batch[j]))/min(k,len(labels_batch[j])) for j in range(len(top_k_classes))])
                print('test_acc: ',tmp_k/20)
                print('#######################')
                print(i,int(total_time),tmp_k/20 , file=out)
                begin_time = time.time()
            idxs_batch, vals_batch, labels_batch = next(training_data_generator)
            sess.run(train_step, feed_dict={x_idxs:idxs_batch, x_vals:vals_batch, y:labels_batch})
            if i%steps_per_epoch==steps_per_epoch-1:
                total_time+=time.time()-begin_time
                print('Finished ',i,' steps. Time elapsed for last 100 batches = ',time.time()-begin_time)
                n_steps_val = n_test//batch_size
                test_data_generator = data_generator_tst(test_files, batch_size)
                num_batches = 0
                p_at_k = 0
                for l in range(n_steps_val): # precision on entire test data
                    idxs_batch, vals_batch, labels_batch = next(test_data_generator)
                    top_k_classes = sess.run(top_idxs, feed_dict={x_idxs:idxs_batch, x_vals:vals_batch})
                    p_at_k += np.mean([len(np.intersect1d(top_k_classes[j],labels_batch[j]))/min(k,len(labels_batch[j])) for j in range(len(top_k_classes))])
                    num_batches += 1
                #
                print('Overall p_at_1 after ',num_batches,'batches = ', p_at_k/num_batches)
                print(i, int(total_time), p_at_k/num_batches, file=out)
                #
                begin_time = time.time()

    writer = tf.summary.FileWriter('logs', sess.graph)
    writer.close()

    # Export the model in Tensorflow v1 SavedModel format
    print('Exporting trained model to ./models')
    builder = tf.compat.v1.saved_model.builder.SavedModelBuilder('./models')

    # Creates the TensorInfo protobuf objects that encapsulates the input/output tensors
    tensor_info_input = tf.compat.v1.saved_model.utils.build_tensor_info(x)

    # output tensor info
    tensor_info_output = tf.compat.v1.saved_model.utils.build_tensor_info(y)

    # Defines the signature, uses the TF Classify API
    amazon_signature = (
        tf.compat.v1.saved_model.signature_def_utils.build_signature_def(
             inputs={'feature_ids_values': tensor_info_input},
             outputs={'classification': tensor_info_output},
             method_name=tf.saved_model.CLASSIFY_METHOD_NAME))

    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.SERVING],
        signature_def_map={
             'slide_amazon':
             amazon_signature,
        })

    # Note: the optimizer is lost... hopefully the function and weights come through!
    builder.save()

    # Save a frozen graphdef too..
#    frozen_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
#         sess,
#         sess.graph_def,
#         ['y']  
#    )

#    output_graph="amazon-frozen-model.pb"
#    with tf.gfile.GFile(output_graph, "wb") as f:
#       f.write(frozen_graph_def.SerializeToString())

    # Try conversion here... 
#    g = tf2onnx.tfonnx.process_tf_graph(sess.graph_def, opset=11, input_names=tensor_info_input, output_names=tensor_info_output)
#    onnx_graph = tf2onnx.optimizer.optimize_graph(g)
#    model_proto = onnx_graph.make_model("converted from {}".format("/tmp/test.onnx"))

    sess.close()

if __name__ == "__main__":
    # execute only if run as a script
    main()
