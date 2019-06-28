'''
This file contains two functions. The first one builds an LSTM RNN model, and the second one used the model to train the parameters and
tests in test set. Before the functions, some variables are defined. These variables can be changed during model evaluation process. This
file can be run directly on terminal line:

python train_test_LSTM.py


Author: Mingchen Li
'''
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import rnn, rnn_cell
import matplotlib.pyplot as plt
import read_data


num_feature = 30
batch_size = 200
rnn_size = 32
output_size = 1
learning_rate = 0.0001

inputs = tf.placeholder('float', [None, 2, num_feature], name = 'inputs')
targets = tf.placeholder('float', name = 'targets')

weight = tf.Variable(tf.truncated_normal([rnn_size, 2]), name = 'weight')
bias = tf.Variable(tf.constant(0.1, shape=[2]),name = 'bias')

training_X, training_Y, dev_X, dev_Y, testing_X, testing_Y = read_data.aline_data('dataset.csv', num_feature)
print(len(training_X))
print(len(dev_X))
print(len(testing_X))


def gru_cell():
   cell = tf.nn.rnn_cell.GRUCell(rnn_size)
   return cell


'''
This function defines a RNN. It is an LSTM RNN for now, but if want to change to GRU, just change the
cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size,state_is_tuple=True)
into
cell = tf.nn.rnn_cell.GRUCell(rnn_size)
'''
def recurrent_neural_network(inputs, w, b):
    layers = 30
    # cell_ = tf.nn.rnn_cell.GRUCell(rnn_size)
    cell = tf.contrib.rnn.MultiRNNCell([gru_cell() for _ in range(layers)])
    att_cell = tf.contrib.rnn.AttentionCellWrapper(cell, attn_length = 30)

    prediction = []
    in_seq = inputs
    ini_state = None
    for _ in range(5):

        outputs, last_State = tf.nn.dynamic_rnn(att_cell, inputs = in_seq, initial_state = ini_state, dtype = tf.float32, scope = "dynamic_rnn")

        outputs = tf.transpose(outputs, [1, 0, 2])
        last_output = tf.gather(outputs, 1, name="last_output")

        pred = tf.matmul(last_output, w) + b

        in_seq = outputs
        ini_state = last_State

        prediction.append(pred)

    return prediction

'''
This function trains the model and tests its performance. After each iteration of the training, it prints out the number of iteration
and the loss of that iteration. When the training is done, prints out the trained parameters. After the testing, it prints out the test
loss and saves the predicted values and the ground truth values into a new .csv file so that it is each to compare the results and
evaluate the model performance. The file has two rows, with the first row being predicted values and second row being real values.
'''
def train_neural_network(inputs):
    
    prediction = recurrent_neural_network(inputs, weight, bias)
    #print(prediction.shape)
    #print(tf.reduce_sum(prediction - targets, 0).shape)
    cost = tf.reduce_sum(tf.square(tf.norm(prediction - targets, ord='euclidean', axis=1)))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_epoch_loss = 1.0
        prev_train_loss = 0.0
        iteration = 0
        train_cost_list = []
        dev_cost_list = []
        while (abs(train_epoch_loss - prev_train_loss) > 1e-5 or iteration < 200):
            iteration += 1
            prev_train_loss = train_epoch_loss
            #train_epoch_loss = 0

            for batch in range(int(len(training_X)/batch_size)):            # There will be some data that's been thrown away if the size of
                                                                            # training_X is not divisible by batch_size
                x_batch, y_batch = read_data.next_batch(batch, batch_size, training_X, training_Y)
                data_feed = {inputs: x_batch, targets: y_batch}
                _, c = sess.run([optimizer, cost], data_feed)
                #print('train: ', c)
                #train_epoch_loss += c/batch_size
            '''
            # training cost
            data_feed = {inputs: training_X, targets: training_Y}
            _, train_c = sess.run([optimizer, cost], data_feed)
            train_epoch_loss = train_c/len(training_X)
            '''
            #train_epoch_loss = train_epoch_loss/int(len(training_X)/batch_size)        # Use the same expression as above to make
                                                                                    # sure not count the data that is thrown away

            dev_epoch_loss = 0
            for batch in range(int(len(dev_X)/batch_size)):
                x_batch, y_batch = read_data.next_batch(batch, batch_size, dev_X, dev_Y)
                data_feed = {inputs: x_batch, targets: y_batch}
                c = sess.run(cost, data_feed)
                #print('dev: ', c)
                dev_epoch_loss += c/batch_size
            dev_epoch_loss = dev_epoch_loss/int(len(dev_X)/batch_size)
            # training cost
            train_epoch_loss = 0
            for batch in range(int(len(training_X)/batch_size)):
                x_batch, y_batch = read_data.next_batch(batch, batch_size, training_X, training_Y)
                data_feed = {inputs: x_batch, targets: y_batch}
                c = sess.run(cost, data_feed)
                #print('dev: ', c)
                train_epoch_loss += c/batch_size
            train_epoch_loss = train_epoch_loss/int(len(training_X)/batch_size)

            # dev cost

            '''
            data_feed = {inputs: dev_X, targets: dev_Y}
            _, dev_c = sess.run([prediction, cost], data_feed)
            dev_epoch_loss = dev_c/len(dev_X)
            '''
            train_cost_list.append(train_epoch_loss)
            dev_cost_list.append(dev_epoch_loss)
            print('Train iteration', iteration,'train loss:',train_epoch_loss)
            print('Train iteration', iteration,'dev loss:',dev_epoch_loss)
        iter_list = range(1, iteration+1)
        plt.figure(1)
        plt.plot(iter_list, train_cost_list)
        plt.plot(iter_list, dev_cost_list)
        plt.title('iteration vs. epoch cost, university')
        plt.show()

        # After the training, print out the trained parameters
        trained_w = sess.run(weight)
        trained_b = sess.run(bias)
        #print('trained_w: ', trained_w, 'trained_b: ', trained_b, 'trained_w shape: ', trained_w.shape)

        # Begin testing
        test_epoch_loss = 0
        test_prediction = np.empty([int(len(testing_X)), 2])
        '''
        data_feed = {inputs: testing_X, targets: testing_Y}
        pre, test_c = sess.run([prediction, cost], data_feed)
        test_prediction = pre
        test_epoch_loss = test_c/int(len(testing_X))
        '''
        test_prediction = np.empty([int(len(testing_X)/batch_size)*batch_size, 2])
        for batch in range(int(len(testing_X)/batch_size)):
            x_batch, y_batch = read_data.next_batch(batch, batch_size, testing_X, testing_Y)
            data_feed = {inputs: x_batch, targets: y_batch}
            pre, c = sess.run([prediction, cost], data_feed)
            pre = np.array(pre)
            test_epoch_loss += c
            test_prediction[batch*batch_size : (batch+1)*batch_size, :] = pre
        test_epoch_loss = test_epoch_loss/(int(len(testing_X)/batch_size)*batch_size)

        print('Test loss:',test_epoch_loss)

        # Save predicted data and ground truth data into a .csv file.
        test_prediction = np.transpose(test_prediction)                                                             # The first row of file: prediction
        testing_Y_array = np.transpose(np.array(testing_Y)[0 : int(len(testing_X)/batch_size)*batch_size, :])       # The second row of file: ground truth
        test_prediction_and_real = np.vstack((test_prediction, testing_Y_array))
        np.savetxt("GRU_test_prediction_and_real.csv", test_prediction_and_real, delimiter = ",")

        # Save model
        saver = tf.train.Saver()
        save_path = saver.save(sess, "./model_weight/model.ckpt")
        print("Model saved.")



        #final test
        x = np.array([[[6.929499999999999549e-01,6.929400000000000004e-01,6.929199999999999804e-01,6.928900000000000059e-01,6.928499999999999659e-01,6.927999999999999714e-01,6.927499999999999769e-01,6.926999999999999824e-01,6.926400000000000334e-01,6.925599999999999534e-01,6.924799999999999844e-01,6.923799999999999955e-01,6.922599999999999865e-01,6.921199999999999575e-01,6.919499999999999540e-01,6.917600000000000415e-01,6.915400000000000436e-01,6.912800000000000056e-01,6.909899999999999931e-01,6.906600000000000517e-01,6.903000000000000247e-01,6.898900000000000032e-01,6.894400000000000528e-01,6.889199999999999768e-01,6.883399999999999519e-01,6.876400000000000290e-01,6.867999999999999661e-01,6.857900000000000107e-01,6.846200000000000063e-01,6.832799999999999985e-01,6.817900000000000071e-01,6.801500000000000323e-01,6.783599999999999630e-01,6.764000000000000012e-01,6.742000000000000215e-01,6.717199999999999838e-01,6.689500000000000446e-01,6.658699999999999619e-01,6.624799999999999578e-01,6.587699999999999667e-01,6.547300000000000342e-01,6.502900000000000347e-01,6.453999999999999737e-01,6.400400000000000533e-01,6.341900000000000315e-01,6.278500000000000192e-01,6.210200000000000164e-01,6.135899999999999688e-01,6.053399999999999892e-01,5.963199999999999612e-01,5.868400000000000283e-01,5.772300000000000209e-01,5.675700000000000189e-01,5.576900000000000190e-01,5.475700000000000012e-01,5.372500000000000053e-01,5.267899999999999805e-01,5.162600000000000522e-01,5.056699999999999529e-01,4.950200000000000156e-01,4.842699999999999783e-01,4.734400000000000275e-01,4.626000000000000112e-01,4.518400000000000194e-01,4.412699999999999956e-01,4.309100000000000152e-01,4.207199999999999829e-01,4.104999999999999760e-01,4.001600000000000157e-01,3.897499999999999853e-01,3.794600000000000195e-01,3.695399999999999796e-01,3.602299999999999947e-01,3.517000000000000126e-01,3.439800000000000080e-01,3.368900000000000228e-01,3.301399999999999890e-01,3.236399999999999832e-01,3.173400000000000110e-01,3.112099999999999866e-01,3.052300000000000013e-01,2.993799999999999795e-01,2.936799999999999966e-01,2.881400000000000072e-01,2.827700000000000213e-01,2.776000000000000134e-01,2.726700000000000235e-01,2.680100000000000260e-01,2.636499999999999955e-01,2.596300000000000274e-01,2.559899999999999953e-01,2.527499999999999747e-01,2.499399999999999955e-01,2.475900000000000045e-01,2.457000000000000017e-01,2.442900000000000071e-01,2.433299999999999907e-01,2.427999999999999881e-01,2.426800000000000068e-01],
                      [5.326100000000000279e-01,5.326400000000000023e-01,5.327600000000000113e-01,5.330300000000000038e-01,5.335100000000000398e-01,5.342200000000000282e-01,5.351799999999999891e-01,5.364100000000000534e-01,5.379099999999999993e-01,5.396699999999999831e-01,5.416800000000000503e-01,5.439399999999999791e-01,5.464200000000000168e-01,5.490899999999999670e-01,5.519500000000000517e-01,5.549600000000000088e-01,5.581000000000000405e-01,5.613599999999999701e-01,5.647199999999999998e-01,5.681699999999999529e-01,5.717200000000000060e-01,5.753899999999999570e-01,5.792000000000000481e-01,5.832000000000000517e-01,5.874899999999999567e-01,5.922600000000000087e-01,5.976299999999999946e-01,6.035700000000000509e-01,6.100200000000000067e-01,6.168500000000000094e-01,6.239099999999999646e-01,6.311200000000000143e-01,6.384400000000000075e-01,6.459500000000000242e-01,6.538000000000000478e-01,6.620399999999999618e-01,6.706400000000000139e-01,6.795299999999999674e-01,6.886299999999999644e-01,6.978799999999999448e-01,7.072599999999999998e-01,7.168299999999999672e-01,7.265899999999999581e-01,7.365099999999999980e-01,7.464899999999999869e-01,7.564400000000000013e-01,7.663200000000000012e-01,7.761500000000000066e-01,7.861000000000000210e-01,7.959300000000000264e-01,8.052399999999999558e-01,8.137199999999999989e-01,8.213399999999999590e-01,8.283000000000000362e-01,8.346200000000000285e-01,8.403000000000000469e-01,8.453000000000000513e-01,8.496399999999999508e-01,8.533199999999999674e-01,8.563899999999999846e-01,8.588700000000000223e-01,8.607700000000000351e-01,8.620999999999999774e-01,8.628799999999999804e-01,8.631299999999999528e-01,8.629099999999999548e-01,8.622499999999999609e-01,8.611600000000000366e-01,8.596300000000000052e-01,8.576700000000000434e-01,8.553399999999999892e-01,8.527099999999999680e-01,8.499100000000000543e-01,8.470800000000000551e-01,8.442899999999999849e-01,8.415500000000000203e-01,8.387700000000000156e-01,8.359499999999999709e-01,8.330800000000000427e-01,8.301500000000000545e-01,8.271800000000000264e-01,8.241500000000000492e-01,8.210899999999999865e-01,8.180100000000000149e-01,8.149300000000000432e-01,8.118800000000000461e-01,8.088800000000000434e-01,8.059800000000000297e-01,8.032000000000000250e-01,8.005799999999999583e-01,7.981599999999999806e-01,7.959699999999999553e-01,7.940500000000000336e-01,7.924099999999999477e-01,7.910899999999999599e-01,7.900899999999999590e-01,7.894099999999999451e-01,7.890399999999999636e-01,7.889500000000000401e-01]]])
        target = np.ones((1,2,1))
        future_step = 30
        x_temp = x[:,:,10:40]
        for i in range(future_step):
            data_feed = {inputs: x_temp}
            pre= sess.run(prediction, data_feed)
            pre = np.expand_dims(np.transpose(np.array(pre)),0)
            x_temp = np.delete(x_temp,0,2)
            x_temp = np.append(x_temp,pre,2)

        print(x_temp,x[:,:,40:70])
        plt.figure(1)
        plt.plot(range(60), np.append(x[0,0,10:40],x_temp[0,0,:]))
        plt.plot(range(60), x[0,0,10:70])
        plt.title('predict vs. truth')
        plt.show()

        plt.figure(1)
        plt.plot(range(60), np.append(x[0,1,10:40],x_temp[0,1,:]))
        plt.plot(range(60), x[0,1,10:70])
        plt.title('predict vs. truth')
        plt.show()


train_neural_network(inputs)


