import tensorflow as tf
import numpy as np

class VariationalAuotoEncoder(object):
    def __init__(self, img_dim=(800,800,3), imu_dim=10, hdim=64, ldim=64, epochs=10000, lr=1e-5):
        self.img_dim = img_dim
        self.imu_dim = imu_dim

        self.seed = 0
        self.epochs = epochs

        self.lr = lr
        self.hdim = hdim
        self.ldim = ldim

        self.std = 1e-1

        self._build_graph()
        self._init_session()

    def _build_graph(self):
        self.g = tf.Graph()
        with self.g.as_default():
            self._placeholders()
            self._vae_nn()
            self._loss_train_op()
            self.init = tf.global_variables_initializer()

    def _placeholders(self):
        # observations, actions and advantages:
        self.img_ph = tf.placeholder(tf.float32, (None,) + self.img_dim, 'obs_img')
        self.imu_ph = tf.placeholder(tf.float32, (None, self.imu_dim), 'obs_imu')

        self.next_img_ph = tf.placeholder(tf.float32, (None,) + self.img_dim, 'next_obs_img')
        self.next_imu_ph = tf.placeholder(tf.float32, (None, self.imu_dim), 'next_obs_imu')

        self.latent_ph = tf.placeholder(tf.float32, (None, self.ldim), 'latent')

        # learning rate:
        self.lr_ph = tf.placeholder(tf.float32, (), 'lr')

    def _vae_nn(self):
        self.latent_mean, self.latent_std = self._encoder(self.img_ph,self.imu_ph)
        self.next_latent_mean, self.next_latent_std = self._encoder(self.next_img_ph,self.next_imu_ph, reuse=True)
        self.latent = self.latent_mean + self.latent_std * tf.random_normal(tf.shape(self.latent_mean))

        self.obs_recon = self._decoder(self.latent)
        self.obs_gen = self._decoder(self.latent_ph, reuse=True)

    def _encoder(self, obs_img, obs_imu, reuse=False):
        hidd_size = self.hdim
        channel_size = self.cdim
        latent_size = self.ldim

        with tf.variable_scope("encoder"):

            out = tf.layers.conv2d(obs_img,filters=channel_size,kernel_size=[20,20],padding="same",activation=tf.nn.relu,name="conv1",
                                   kernel_initializer=tf.random_normal_initializer(stddev=0.01),reuse=reuse)
            out = tf.layers.max_pooling2d(out,pool_size=[2,2],strides=2,name="pool1",reuse=reuse)

            out = tf.layers.conv2d(out,filters=channel_size,kernel_size=[20,20],padding="same",activation=tf.nn.relu,name="conv2",
                                   kernel_initializer=tf.random_normal_initializer(stddev=0.01),reuse=reuse)
            out = tf.layers.max_pooling2d(out,pool_size=[4,4],strides=4,name="pool2",reuse=reuse)

            out = tf.layers.conv2d(out,filters=channel_size,kernel_size=[20,20],padding="same",activation=tf.nn.relu,name="conv3",
                                   kernel_initializer=tf.random_normal_initializer(stddev=0.01),reuse=reuse)
            out = tf.layers.max_pooling2d(out,pool_size=[100,100],strides=1,name="pool3",reuse=reuse)
            z_img = tf.layers.flatten(out,name="img_latent",reuse=reuse)

            out = tf.concat([obs_imu,z_img],axis=1)
            out = tf.layers.dense(out, hidd_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01, seed=self.seed),
                                  name="h1",
                                  reuse=reuse)
            out = tf.layers.dense(out, hidd_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01, seed=self.seed),
                                  name="h2",
                                  reuse=reuse)
            latent_mean = tf.layers.dense(out, latent_size,
                                          kernel_initializer=tf.random_normal_initializer(stddev=0.01, seed=self.seed),
                                          name="latent_mean", reuse=reuse)
            latent_std_logits = tf.layers.dense(out, latent_size,
                                                kernel_initializer=tf.random_normal_initializer(stddev=0.01,
                                                                                                seed=self.seed),
                                                name="latent_std", reuse=reuse)
            latent_std = self.std * tf.sigmoid(latent_std_logits)
        return latent_mean, latent_std

    def _decoder(self, latent, reuse=False):
        hidd_size = self.hdim
        channel_size = self.cdim

        with tf.variable_scope("decoder"):
            # Decoder for next state
            out = tf.layers.dense(latent, hidd_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01, seed=self.seed),
                                  name="h1",
                                  reuse=reuse)
            out = tf.layers.dense(out, hidd_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01, seed=self.seed),
                                  name="h2",
                                  reuse=reuse)
            out = tf.layers.dense(out, self.imu_dim + channel_size,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01, seed=self.seed),
                                  name="obs", reuse=reuse)
            obs_imu = out[:,:self.imu_dim]
            z_img = out[:,self.imu_dim:]
            out = tf.layers.conv2d_transpose(z_img,filters=channel_size,kernel_size=[20,20],padding="same",activation=tf.nn.relu,name="deconv1",
                                   kernel_initializer=tf.random_normal_initializer(stddev=0.01),reuse=reuse)
            out = tf.layers.conv2d_transpose(z_img,filters=channel_size,kernel_size=[20,20],padding="same",activation=tf.nn.relu,name="deconv1",
                                   kernel_initializer=tf.random_normal_initializer(stddev=0.01),reuse=reuse)
            out = tf.layers.conv2d_transpose(z_img,filters=1,kernel_size=[20,20],padding="same",activation=tf.nn.relu,name="deconv1",
                                   kernel_initializer=tf.random_normal_initializer(stddev=0.01),reuse=reuse)


        return obs

    def _loss_train_op(self):
        reconstruction_loss = tf.reduce_sum(tf.squared_difference(self.obs_recon, self.obs_ph))
        kl_dist_loss = -0.5 * tf.reduce_sum(1 + 2 * tf.log(self.latent_std) - 2 * tf.log(self.next_latent_std) \
                                            - tf.square(
            (self.latent_mean - self.next_latent_mean) / self.next_latent_std) - tf.square(
            self.latent_std / self.next_latent_std), axis=1)
        kl_prior_loss = -0.5 * tf.reduce_sum(1 + 2 * tf.log(self.latent_std) \
                                             - tf.square(self.latent_mean) - tf.square(self.latent_std), axis=1)

        self.reconstruction_loss = tf.reduce_mean(reconstruction_loss)
        self.kl_dist_loss = tf.reduce_mean(kl_dist_loss)
        self.kl_prior_loss = tf.reduce_mean(kl_prior_loss)

        self.vae_loss = self.reconstruction_loss + self.kl_dist_loss + 1e-3 * self.kl_prior_loss

        # OPTIMIZER
        optimizer = tf.train.AdamOptimizer(self.lr_ph)
        self.train_op = optimizer.minimize(self.vae_loss)

    def _init_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config, graph=self.g)
        self.sess.run(self.init)

    def encode(self, obs):
        return self.sess.run(self.latent_mean, feed_dict={self.obs_ph: obs})

    def decode(self, latent):
        return self.sess.run(self.obs_gen, feed_dict={self.latent_ph: latent})

    def train(self, obs, next_obs, batch_size=128):

        num_batches = max(obs.shape[0] // batch_size, 1)
        batch_size = obs.shape[0] // num_batches

        for e in range(self.epochs + 1):
            obs, next_obs = shuffle(obs, next_obs, random_state=0)
            for j in range(num_batches):
                start = j * batch_size
                end = (j + 1) * batch_size
                feed_dict = {self.obs_ph: obs[start:end, :],
                             self.next_obs_ph: next_obs[start:end, :],
                             self.lr_ph: self.lr}
                self.sess.run(self.train_op, feed_dict)

            if (e % 500) == 0:
                feed_dict = {self.obs_ph: obs, self.next_obs_ph: next_obs, self.lr_ph: self.lr}
                vae_loss, kl_dist_loss, kl_prior_loss = self.sess.run(
                    [self.vae_loss, self.kl_dist_loss, self.kl_prior_loss], feed_dict)
                print('[{:05d}/{:05d}] Recon : {:.03f}, KL Dist : {:.03f}, KL Prior : {:.03f}'.format(e, self.epochs,
                                                                                                      vae_loss,
                                                                                                      kl_dist_loss,
                                                                                                      kl_prior_loss))
        return vae_loss