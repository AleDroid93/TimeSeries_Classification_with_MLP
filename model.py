import tensorflow as tf

class MyMultilayerPerceptron(tf.keras.Model):
    def __init__(self, n_classes, dropout_rate = 0.0, units=32, hidden_activation='relu', output_activation = 'softmax',
                 name='mlpNetwork', **kwargs):
        super(MyMultilayerPerceptron, self).__init__()
        self.hidden1 = tf.keras.layers.Dense(units=units, activation=hidden_activation)
        self.batch1 = tf.keras.layers.BatchNormalization()
        self.dropout1 = tf.keras.layers.Dropout(rate=dropout_rate)
        self.hidden2 = tf.keras.layers.Dense(units=units, activation=hidden_activation)
        self.batch2 = tf.keras.layers.BatchNormalization()
        self.dropout2 = tf.keras.layers.Dropout(rate=dropout_rate)
        self.model_out = tf.keras.layers.Dense(units=n_classes, activation=output_activation)



    def call(self, inputs, training=False):
        inputs = self.hidden1(inputs)
        inputs = self.batch1(inputs, training = training)
        inputs = self.dropout1(inputs, training = training)
        inputs = self.hidden2(inputs)
        inputs = self.batch2(inputs, training=training)
        inputs = self.dropout2(inputs, training=training)
        return self.model_out(inputs)
