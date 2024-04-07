

import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, TimeDistributed, Dense, Masking, concatenate, GRU
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import pickle
import os
import keras
  
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

class Detect:
    ntimesteps=656
    classes = []
    @classmethod
    def pad(cls, x_data):
        for ind in range(len(x_data)):
            x_data[ind] = np.pad(x_data[ind], ((0, cls.ntimesteps - len(x_data[ind])), (0, 0)))
        return x_data

    @classmethod
    def init(cls):
        pretrained_model = os.path.join(SCRIPT_DIR, "pretrained.keras")
        iforests = os.path.join(SCRIPT_DIR, "iforests.pickle")
        cls.mod = Custom(656, 4, 2, 9, 12)
        model = keras.models.load_model(pretrained_model)
        cls.mod.custom_model(model, 'lc', 'host', 'latent')
        cls.mod.create_encoder()
        cls.mod.mcif = mcif()
        with open(iforests, 'rb') as f:
            cls.mod.mcif.iforests = pickle.load(f)

        
        # cls.mod = Custom(ntimesteps, 4, 2, 9, y_train.shape[-1])
        # cls.mod.custom_model(best.model, 'lc', 'host', 'latent')
        # cls.mod.create_encoder()
        # cls.mod.mcif = mcif()
        # cls.mod.mcif.iforests = best.iso_forests

    @classmethod
    def predict(cls, x_data, host_gal):
        return cls.mod.predict(x_data, host_gal)

    @classmethod
    def anomaly_score(cls, x_data, host_gal):
        return cls.mod.score(x_data, host_gal)

    @classmethod
    def plot_real_time(cls, x_data, host_gal):
        cls.mod.plot_real_time(x_data, [0.4827, 0.6223], x_data[:, 1] * 100 - 30, x_data[:, 2] * 500, x_data[:, 3] * 500, host_gal, colors=['red', 'g'], names=['r', 'g'])
    
        
    

class Custom:
    def __init__(self, timesteps, features, contextual, latent_size, n_classes):
        self.n_classes=n_classes
        self.features=features
        self.contextual=contextual
        self.latent_size = latent_size
        self.timesteps=timesteps

    def pad(self, x_data):
        for ind in range(len(x_data)):
            x_data[ind]=np.pad(x_data[ind], ((0, self.timesteps - len(x_data[ind])), (0, 0)))
        return x_data

    def create_model(self):
        input_1 = Input((self.timesteps, self.features), name='lc')  # X.shape = (Nobjects, Ntimesteps, 4) CHANGE
        self.lc_name = 'lc'
        masking_input1 = Masking(mask_value=0.)(input_1)
    
        lstm1 = GRU(100, return_sequences=True, activation='tanh')(masking_input1)
        lstm2 = GRU(100, return_sequences=False, activation='tanh')(lstm1)
    
        dense1 = Dense(100, activation='tanh')(lstm2)

        if (self.contextual > 0):
            input_2 = Input(shape = (self.contextual, ), name='host') # CHANGE
            self.context_name = 'host'
            dense2 = Dense(10)(input_2)
            merge1 = concatenate([dense1, dense2])

        else:
            merge1 = dense1
    
        dense3 = Dense(100, activation='relu')(merge1)
    
        dense4 = Dense(self.latent_size, activation='relu', name='latent')(dense3)
    
        output = Dense(self.n_classes, activation='softmax')(dense4)

        if (self.contextual):
            self.model = Model(inputs=[input_1, input_2], outputs=output)
        else:
            self.model = Model(inputs=[input_1], outputs=output)
    
    
        self.model.compile(loss = "categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
        
        self.latent_name='latent'
        
    def custom_model(self, model, lc_name, context_name, latent_name=None):
        self.model=model
        self.lc_name = lc_name
        self.context_name = context_name
        self.latent_name = latent_name
        
    def train(self, X_train, y_train, X_val, y_val, host_gal_train = None, host_gal_val = None, class_weights=None):
        
        early_stopping = EarlyStopping(
                                      patience=5,
                                      min_delta=0.001,                               
                                      monitor="val_loss",
                                      restore_best_weights=True
                                      )
        
        
        if (self.contextual > 0):
            self.history = self.model.fit(x = [X_train, host_gal_train], validation_data=([X_val, host_gal_val], y_val), y = y_train, epochs=40, batch_size = 128, class_weight = class_weights, callbacks=[early_stopping])
        else:
            self.history = self.model.fit(x = [X_train], validation_data=([X_val], y_val), y = y_train, epochs=40, batch_size = 128, class_weight = class_weights, callbacks=[early_stopping])
 

        
    
    def create_encoder(self):
        if (self.contextual):
            self.latent_model = Model(inputs=[self.model.get_layer(self.lc_name).input, self.model.get_layer(self.context_name).input], outputs=self.model.get_layer(self.latent_name).output)
        else:
            self.latent_model = Model(inputs=[self.model.get_layer(self.lc_name).input], outputs=self.model.get_layer(self.latent_name).output)
            

    def predict(self, x_data, host_gal=None):
        if (self.contextual > 0):
            return self.model.predict(x = [x_data, host_gal])
        else:
            return self.model.predict(x = [x_data])
        
        
    def encode(self, x_data, host_gal=None):
        if (self.contextual > 0):
            return self.latent_model.predict(x = [x_data, host_gal])
        else:
            return self.latent_model.predict(x = [x_data])
    
    def init_mcif(self, x_data, y_data, host_gal, n_estimators=100):
        self.mcif = mcif(n_estimators)
        self.mcif.train(self.encode(x_data, host_gal), y_data)
        
        

    def score(self, x_data, host_gal=None):
        return self.mcif.score(self.encode(x_data, host_gal))

    def get_anomaly_real_time(self, curves, host_galaxy=None):
        
        splits = []
        lcs = []
        host_gals = []
        for ind in range(len(curves)):
            cur = np.zeros((self.timesteps, 4))
            anomaly_scores = []
            if (self.contextual):
                host_gal = np.array(host_galaxy[ind])
            
            curve = curves[ind]
            
            for ind, i in enumerate(curve):
                if (np.count_nonzero(i) == 0):
                    break
                cur[ind]=i
    
                lcs.append(cur.copy())
                if (self.contextual):
                    host_gals.append(host_gal)
        
            splits.append(len(lcs))
    
        lcs = np.array(lcs)
        host_gals = np.array(host_gals)
    
        scores = self.score(np.array(lcs), np.array(host_gals))
        
        ans = []
        prv=0
        for diff in splits:
            ans.append(scores[prv:diff])
            prv=diff
        
        return ans


    def plot_real_time(self, x_data, bands, time_, flux_, error_, host_gal=None, names = [], colors = []):
        classification_scores = self.get_anomaly_real_time([x_data], [host_gal])[0]
        tot = len(classification_scores)
        time_ = time_[:tot]
        flux_ = flux_[:tot]
        error_ = error_[:tot]

        assert(len(classification_scores) == len(time_));

        time = {i : [] for i in bands}
        flux = {i : [] for i in bands}
        error = {i : [] for i in bands}

        
    
        for ind, i in enumerate(x_data[:len(classification_scores)]):
            if (not np.any(i)):
                break
            flux[i[0]].append(flux_[ind])
            error[i[0]].append(error_[ind])
            time[i[0]].append(time_[ind])
    
        fig, axs = plt.subplots(2, figsize=(10, 20))
    
        plt.subplots_adjust(wspace=0, hspace=0)
    
        axs[0].set_title(f"Real Time Anomaly Score", fontsize=30)
    
    
        axs[0].set_ylabel('Flux', fontsize=27)
        for ind, i in enumerate(bands):
            axs[0].errorbar(time[i], flux[i], yerr=error[i], fmt='.', label = names[ind] if len(names) else None, color = colors[ind] if len(colors) else None)
    
    
        axs[1].set_ylabel('Anomaly Score', fontsize=27)
        axs[1].set_xlabel('Time Since Trigger', fontsize=27)
        axs[1].plot(time_, classification_scores)
    
        axs[1].set_ylim(-0.3, 0.3)
        axs[1].set_yticks(ticks=np.arange(-0.3, 0.3, 0.1))
    
        axs[0].tick_params(axis='both', labelsize=27)
        axs[1].tick_params(axis='both', labelsize=27)
        
        axs[0].legend()

        plt.show()
        
        

class mcif:
    def __init__(self, n_estimators = 100):
        self.n_estimators=n_estimators

    def train(self, x_data, labels):
        self.classes = np.unique(labels, axis=0)
        self.iforests = [IsolationForest(n_estimators=self.n_estimators) for i in self.classes]
        
        for ind, cls in enumerate(self.classes):
            here = []
            for i in range(len(x_data)):
                if (list(cls) == list(labels[i])):
                    here.append(x_data[i])

            self.iforests[ind].fit(here)
            

    def score_discrete(self, data):
        scores = [-det.decision_function(data) for det in self.iforests]

        scores = np.array(scores)
        scores = scores.T

        return scores

    def score(self, data):
        return [np.min(i) for i in self.score_discrete(data)]