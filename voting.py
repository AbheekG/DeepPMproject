from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction=0.3
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# from tensorflow import keras

import keras
import keras.backend as K
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization
from keras.layers import Lambda, Multiply, Add, Maximum, Average, Concatenate
# from keras.datasets import mnist
# from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU, Softmax
# from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt

# import sys

import numpy as np

class VoterNet():
    def __init__(self, n=3, m=2):

        # assert (constraint == "barrier" or constraint == "penalty" 
        #     or constraint == 'lagrangian')
        # self.constraint = constraint

        self.n_agents = n
        self.n_alters = m
        self.vote_shape = (self.n_agents, self.n_alters)
        
        self.latent_dim = 10*m      # For GAN

        self.models = {}

        # build combined_generator, combined_mechanism for GAN
        self.build_gan()
        
        # print("\n\n\nGAN generator summary.")
        # self.make_weights_trainable(model="generator")
        # self.models['gan_generator'].summary()

        # print("\n\n\nGAN mechanism summary.")
        # self.make_weights_trainable(model="mechanism")
        # self.models['gan_mechanism'].summary()

    def build_gan(self):
        
        optimizer = Adam(0.0002, 0.5)

        z = Input(shape=(self.latent_dim,))

        # Build the generator
        self.vote_generator = self.build_vote_generator()
        self.strategic_vote_generator = self.build_strategic_vote_generator()

        # The generator takes noise as input and generates votes.
        votes = self.vote_generator(z)
        # all_votes = [true_votes, voter_softmax, strategic_votes]
        all_votes = self.strategic_vote_generator(votes)

        # Mechanism, used with or without generator.
        self.mechanism = self.build_mechanism()
        # For the combined model we will only train the generator
        self.mechanism.trainable = False

        # Compute the probability_output for all the inputs.
        # Doing it here because don't want to pass the mechanism to somewhere else.
        probs = []
        probs.append ( self.mechanism(all_votes[0]) )
        probs.append ( all_votes[1] );
        for i in range(self.n_agents):
            probs.append(self.mechanism(all_votes[i+2]))


        # Building models...
        # Building the mechanism model with only mechanism trainable.
        self.make_weights_trainable(model="mechanism")

        # Distortion
        dist = Lambda(self.gan_distortion_layer)(all_votes + probs)
        
        self.models['gan_mech_dist'] = Model(z, dist)
        self.models['gan_mech_dist'].compile(
            loss=self.mean_pred_loss, optimizer=optimizer)

        # Regret
        regt = Lambda(self.regret_layer)(all_votes + probs)
        regt_max = Lambda(lambda x: K.max(x, axis=-1))(regt)

        self.models['gan_mech_regt_lmd'] = Model(z, regt)
        self.models['gan_mech_regt_lmd'].compile(
            loss=self.mean_pred_loss, optimizer=optimizer)

        self.models['gan_mech_regt'] = Model(z, regt_max)
        self.models['gan_mech_regt'].compile(
            loss=self.mean_pred_loss, optimizer=optimizer)

        # Penalty Loss.
        rho = Input(shape=(1,))
        loss_penalty = Lambda(self.penalty_loss)([dist, regt, rho])

        self.models['gan_mech_penalty'] = Model([z, rho], loss_penalty)
        self.models['gan_mech_penalty'].compile(loss=self.mean_pred_loss, optimizer=optimizer)

        # Augmented Lagrangian.
        lmd = Input(shape=(self.n_agents,))
        loss_auglag = Lambda(self.auglag_loss)([dist, regt, rho, lmd])
        
        self.models['gan_mech_auglag'] = Model([z, rho, lmd], loss_auglag)
        self.models['gan_mech_auglag'].compile(loss=self.mean_pred_loss, optimizer=optimizer)


        # Now building the same model, but the generator is trainable
        self.make_weights_trainable(model="generator")

        dist_neg = Lambda(lambda x: -1. * x )(dist)
        self.models['gan_gene_dist'] = Model(z, dist_neg)
        self.models['gan_gene_dist'].compile(loss=self.mean_pred_loss, optimizer=optimizer)
        
        regt_max_neg = Lambda(lambda x: -1. * x )(regt_max)
        self.models['gan_gene_regt'] = Model(z, regt_max_neg)
        self.models['gan_gene_regt'].compile(loss=self.mean_pred_loss, optimizer=optimizer)
        
        loss_penalty_neg = Lambda(lambda x: -1. * x )(loss_penalty)
        self.models['gan_gene_penalty'] = Model([z, rho], loss_penalty_neg)
        self.models['gan_gene_penalty'].compile(loss=self.mean_pred_loss, optimizer=optimizer)
        
        loss_auglag_neg = Lambda(lambda x: -1. * x )(loss_auglag)
        self.models['gan_gene_auglag'] = Model([z, rho, lmd], loss_auglag_neg)
        self.models['gan_gene_auglag'].compile(loss=self.mean_pred_loss, optimizer=optimizer)


    def make_weights_trainable(self, model):
        if model == "generator":
            self.mechanism.trainable = False
            self.vote_generator.trainable = True
            self.strategic_vote_generator.trainable = True
        elif model == "mechanism":
            self.mechanism.trainable = True
            self.vote_generator.trainable = False
            self.strategic_vote_generator.trainable = False
        else:
            assert(0 and "Failed to make_weights_trainable")

    def mean_pred_loss(self, y_true, y_pred):
        return K.mean(y_pred)

    def distortion(self, tensors):
        votes_ = tensors[0]
        probs_ = tensors[1]
        welfares = K.sum(votes_, axis=-2)
        expected_welfare = K.sum(Multiply()([welfares, probs_]), axis=-1)
        max_welfare = K.max(welfares, axis=-1)
        return (max_welfare / expected_welfare)


    def gan_distortion_layer(self, tensors):
        """
        all_votes = [true_votes, voter_softmax, strategic_votes]
        probs = [true_probs, voter_softmax, strategic_probs]
        tensors = all_votes + probs
        """
        losses = []
        for i in range(self.n_agents+2):
            if i == 1:
                continue
            votes_ = tensors[i]
            probs_ = tensors[self.n_agents+2 + i]
            losses.append(self.distortion([votes_, probs_]))

        return Maximum()(losses)


    def regret_layer(self, tensors):
        """
        all_votes = [true_votes, voter_softmax, strategic_votes]
        probs = [true_probs, voter_softmax, strategic_probs]
        tensors = all_votes + probs

        strategy proofness:
        vote1 * prob(vote1) >= vote1 * prob(vote2)
        c >= 0
        penalty:
            loss = max(0, -c)
            loss = max(0, vote1 * prob(vote2) - vote1 * prob(vote1))^2
        barrier:
            loss = -log(c)
            loss = -log(vote1 * prob(vote1) - vote1 * prob(vote2))
            adding additional Relu to prevent nexgative log
        """
        losses = []
        for i in range(self.n_agents):
            vote1 = tensors[0][:,i,:]
            prob1 = tensors[self.n_agents+2]
            vote2 = tensors[i+2][:,i,:]
            prob2 = tensors[i+2+self.n_agents+2]
            losses.append ( K.square( Multiply()([
                  K.relu( K.sum(Multiply()([vote1, prob2]), axis=-1)
                        - K.sum(Multiply()([vote1, prob1]), axis=-1) )
                + K.relu( K.sum(Multiply()([vote2, prob1]), axis=-1)
                        - K.sum(Multiply()([vote2, prob2]), axis=-1) )
                                    , tensors[1][:,i] ] ) ) )
            # print(losses[-1].shape)

        if len(losses) > 1:
            return Concatenate()(losses)
        else:
            return losses[0]

    def penalty_loss(self, tensors):
        """
        tensor = [dist, regt, rho]
        loss = dist + rho * sum[regt^2]
        """
        dist_ = tensors[0]
        regt_ = tensors[1]
        rho_ = tensors[2]

        loss = dist_ + rho_ * K.sum(Multiply()([regt_, regt_]), axis=-1)

        return loss

    def auglag_loss(self, tensors):
        """
        tensor = [dist, regt, rho, lmd]
        loss = dist + 0.5 * rho * sum[regt^2] + sum[regt*lmd]
        """
        dist_ = tensors[0]
        regt_ = tensors[1]
        rho_ = tensors[2]
        lmd_ = tensors[3]

        # print(regt_.shape, lmd_.shape)

        loss = dist_ + 0.5 * rho_ * K.sum(Multiply()(
            [regt_, regt_]), axis=-1) + K.sum(Multiply()([regt_, lmd_]), axis=-1)

        return loss

    def build_vote_generator(self):

        model = Sequential()

        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.vote_shape)))
        model.add(Reshape(self.vote_shape))
        model.add(Softmax(axis=-1))

        print ("build_vote_generator model summary")
        model.summary()

        noise = Input(shape=(self.latent_dim,))
        votes = model(noise)

        return Model(noise, votes)

    def build_strategic_vote_generator(self):

        model = Sequential()

        model.add(Flatten(input_shape=self.vote_shape))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        print ("\n\n\nStrategic_vote_generator base model summary")
        model.summary()

        model1 = Sequential()
        model1.add(Dense(256, input_dim=256))
        model1.add(LeakyReLU(alpha=0.2))
        model1.add(BatchNormalization(momentum=0.8))
        model1.add(Dense(self.n_alters))
        model1.add(Reshape((1, self.n_alters)))
        model1.add(Softmax(axis=-1))
        
        print ("\n\n\nStrategic_vote_generator derived model1 summary")
        model1.summary()


        model2 = Sequential()
        model2.add(Dense(256, input_dim=256))
        model2.add(LeakyReLU(alpha=0.2))
        model2.add(BatchNormalization(momentum=0.8))
        model2.add(Dense(self.n_agents))
        model2.add(Softmax(axis=-1))

        print ("\n\n\nStrategic_vote_generator derived model2 summary")
        model2.summary()

        votes = Input(shape=self.vote_shape)
        intermediate_votes = model(votes)
        # Gets the strategic vote.
        strategic_vote = model1(intermediate_votes)
        # Gets the probability dist for voter.
        strategic_voter = model2(intermediate_votes)

        # Custom layer for strategic vote generator
        def strategic_vote_generator_layer(tensors):
            votes = tensors[0]
            strategic_vote = tensors[1]
            strategic_votes = []
            for i in range(self.n_agents):
                strategic_votes.append(K.concatenate([votes[:, :i, :], 
                            strategic_vote, votes[:, i+1:, :]], axis=1))
                # print(strategic_votes[i].shape)
            # Seperating the votes by the weights.
            all_votes = [votes] + [tensors[2]] + strategic_votes
            return all_votes

        def strategic_vote_generator_layer_output_shape(input_shapes):
            return [input_shapes[0]] + [input_shapes[2]] + [input_shapes[0]]*self.n_agents

        layer = Lambda(strategic_vote_generator_layer,
            strategic_vote_generator_layer_output_shape)
        all_votes = layer([votes, strategic_vote, strategic_voter])

        final_model = Model(votes, all_votes)
        print ("\n\n\nStrategic_vote_generator final model summary")
        final_model.summary()

        return final_model

    def build_mechanism(self):

        model = Sequential()

        model.add(Flatten(input_shape=self.vote_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(self.n_alters))
        model.add(Softmax(axis=-1))

        print ("\n\n\nMechanism model summary")
        model.summary()

        vote = Input(shape=self.vote_shape)
        winner = model(vote)

        return Model(vote, winner)

    def train_gan(self, epochs, sub_iters=32, batch_size=128, sample_interval=50,
        loss="penalty", model="both", param_epochs=10, params=[1]):

        assert (loss == "penalty" or loss == "auglag" or loss == "dist" or loss == "regt")
        assert (model == "both" or model == "mech" or model == "gene")

        model_mech = self.models['gan_mech_' + loss]
        model_gene = self.models['gan_gene_' + loss]

        # Dummy output because I don't know how to create loss-function without that
        dummy_output = np.zeros((batch_size, 1))

        m_loss = 0
        g_loss = 0
        dist = 0
        regt = 0
        dist_old = 0
        dist_vel = 0

        # For loss
        if loss == "penalty" or loss == "auglag":
            rho = params[0] * np.ones((batch_size, 1))
            lmd = np.random.normal(0, 1, (batch_size, self.n_agents))

        for epoch in range(epochs):
            print ("Epoch %d. rho = %f. dist_vel = %f." % (epoch, rho.max(), dist_vel))

            # Generate Noise.
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            
            if epoch % param_epochs and np.abs(dist_vel) < 0.001:
                # The second condition to make sure that veltocity stopped updating
                print ("Updating rho and lmd")
                if rho.max() > 10**5:
                    print("rho = %f, too large", rho.max())

                # if regt > 10**10:
                #     print("rho = %f, too large", rho)

                if (loss == "penalty" or loss == "auglag") and rho.max() <= 10**5:
                    rho *= 2
                if loss == "auglag":
                    lmd += rho * self.models['gan_mech_regt_lmd'].predict(noise).mean(axis=0)
                    
            print (lmd.shape)
            dummy_input = noise
            if loss == "penalty" or loss == "auglag":
                dummy_input = [noise, rho]
            if loss == "auglag": dummy_input.append(lmd)
            # Train the mechanism
            self.make_weights_trainable(model="mechanism")
            for i in range(sub_iters):
                if model == "mech" or model == "both":
                    m_loss = model_mech.train_on_batch(dummy_input, dummy_output)
                    print ("%d (after mechanism) [M loss: %f] [G loss: %f]" % (epoch, m_loss, g_loss))
            dist = self.models['gan_mech_dist'].predict(noise).mean()
            regt = self.models['gan_mech_regt'].predict(noise).max()
            print ("%d [Distortion: %f] [Regret: %f]" % (epoch, dist, regt))


            # Updating the velocity for loss.
            dist_vel *= (1 - 0.01/param_epochs)
            dist_vel += (0.01/param_epochs) * (dist - dist_old)
            dist_old = dist

            # Train Generator
            self.make_weights_trainable(model="generator")
            for i in range(sub_iters):
                if model == "gene" or model == "both": 
                    g_loss = model_gene.train_on_batch(dummy_input, dummy_output)
                    print ("%d (after generator) [M loss: %f] [G loss: %f]" % (epoch, m_loss, g_loss))

            # dist = self.models['gan_mech_dist'].predict(noise)
            # regt = self.models['gan_mech_regt'].predict(noise)
            # print ("%d       (after generator) [Distortion: %f] [Regret: %f]" % (epoch, dist, regt))


            # If at save interval => save weights.
            if epoch % sample_interval == 0:
                model_mech.save_weights("weights/n-%d_m-%d_loss-%s_model-%s_epoch-%d.h5" % (
                    self.n_agents, self.n_alters, loss, model, epoch))
                print('Saved "weights/n-%d_m-%d_loss-%s_model-%s_epoch-%d.h5"' % (
                    self.n_agents, self.n_alters, loss, model, epoch))

    def random_noise(self, k = 10):
        return np.random.normal(0, 1, (k, self.latent_dim))

    def random_votes(self, k = 10):
        votes = np.random.uniform(0, 1, (k, self.n_agents, self.n_alters))
        votes = votes / votes.sum(axis=-1, keepdims=True)
        return votes


if __name__ == '__main__':



    gan = VoterNet(n=3,m=2)
    gan.train_gan(epochs=2001, batch_size=128, sub_iters=16, sample_interval=100)
    noise = gan.random_noise(10)
    votes = gan.random_votes(10)
    gan.mechanism.predict(votes)

    gan.models['gan_mech_dist'].predict(noise)
    gan.models['gan_mech_regt'].predict(noise)

    # gan.train_gan(epochs=201, batch_size=512, sub_iters=32, sample_interval=100, loss='auglag', model='mech')


    # (votes.sum(-2).max(-1) / (votes.sum(-2) *  gan.mechanism.predict(votes)).sum(-1)).mean()


    # votes.argmax(-1).sum(-1)

