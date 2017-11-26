import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.misc
import os
import csv
import itertools
import tensorflow.contrib.slim as slim
#imageio.plugins.ffmpeg.download()

# This is a simple function to reshape our game frames.
def processState(state1):
    return np.reshape(state1, [21168])

# This is a simple function to reshape our game frames.
def processImage(frame, x):
    s = frame[10:-10,30:-30]
    if len(frame.shape) == 2:
        s = scipy.misc.imresize(s,[x,x])
    else:
        s = scipy.misc.imresize(s,[x,x,frame.shape[2]])
    s = np.reshape(s,[np.prod(s.shape)])
    return s

# This is a simple function to reshape our game buffers and create a N channel image, N being the number of images passed to the function
def processBuffers(image_size, d, l, s):
    d = d[10:-10,30:-30]
    d = scipy.misc.imresize(d,[image_size, image_size])
    l = l[10:-10,30:-30]
    l = scipy.misc.imresize(l,[image_size, image_size])
    s = s[10:-10,30:-30]
    s = scipy.misc.imresize(s,[image_size, image_size])
    dls = np.array([d,l,s])
    dls = np.rollaxis(dls, 0, 3)
    dls = np.reshape(dls,[np.prod(dls.shape)])
    return dls

# These functions allows us to update the parameters of our target network with those of the primary network.
def updateTargetGraph(tfVars, tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx, var in enumerate(tfVars[0:total_vars // 2]):
        op_holder.append(tfVars[idx + total_vars // 2].assign(
            (var.value() * tau) + ((1 - tau) * tfVars[idx + total_vars // 2].value())))
    return op_holder


def updateTarget(op_holder, sess):
    for op in op_holder:
        sess.run(op)
    total_vars = len(tf.trainable_variables())
    a = tf.trainable_variables()[0].eval(session=sess)
    b = tf.trainable_variables()[total_vars // 2].eval(session=sess)
    if a.all() == b.all():
        print("Target Set Success")
    else:
        print("Target Set Failed")


# Record performance metrics and episode logs for the Control Center.
def saveToCenter(i, rList, jList, bufferArray, summaryLength, h_size, sess, mainQN, time_per_step, img_x, img_z, chls, path):
    with open(path + '/log.csv', 'a') as myfile:
        state_display = (np.zeros([1, h_size]), np.zeros([1, h_size]))
        imagesS = []
        '''
        for idx, z in enumerate(np.vstack(bufferArray[:, 0])):
            img, state_display = sess.run([mainQN.salience, mainQN.rnn_state],
                                          feed_dict={
                                              mainQN.scalarInput: np.reshape(bufferArray[idx, 0], [1, img_x*img_x*img_z]) / 255.0, \
                                              mainQN.trainLength: 1, mainQN.state_in: state_display,
                                              mainQN.batch_size: 1})
            imagesS.append(img)
        
        imagesS = (imagesS - np.min(imagesS)) / (np.max(imagesS) - np.min(imagesS))
        imagesS = np.vstack(imagesS)
        imagesS = np.resize(imagesS, [len(imagesS), img_x, img_x, img_z])
        luminance = np.max(imagesS,  img_z)
        imagesS = np.multiply(np.ones([len(imagesS),  img_x,  img_x,  img_z]), np.reshape(luminance, [len(imagesS),  img_x, img_x, 1]))
        make_gif(np.ones([len(imagesS), img_x, img_x, img_z]), './Center/frames/sal' + str(i) + '.gif',
                 duration=len(imagesS) * time_per_step, true_image=False, salience=True, salIMGS=luminance)
        '''
        #print(bufferArray[:, 0])
        images = list(zip(bufferArray[:, 0]))
        images.append(bufferArray[-1, 3])
        #print(images)
        images = np.vstack(images)
        images = np.resize(images, [len(images), img_x, img_x, img_z])
        
        #When we have depth and label buffer stacks, just keep depth buffer to display
        if img_z > 1:
            
            d = np.array([images[...,0]])
            d = np.rollaxis(d, 0, 4)
            l = np.array([images[...,1]])
            l = np.rollaxis(l, 0, 4)
            s = np.array([images[...,2]])
            s = np.rollaxis(s, 0, 4)

            
            make_gif(l, path + '/frames/l/labels' + str(i) + '.gif',
                     duration=len(images) * time_per_step,
                     true_image=True, salience=False)
            
            make_gif(d, path + '/frames/d/depth' + str(i) + '.gif',
                     duration=len(images) * time_per_step,
                     true_image=True, salience=False)

            make_gif(s, path + '/frames/image' + str(i) + '.gif',
                     duration=len(images) * time_per_step,
                     true_image=True, salience=False)
        else:
            s = np.array([images[...,0]])
            s = np.rollaxis(s, 0, 4)
            
            make_gif(s, path + '/frames/image' + str(i) + '.gif',
                     duration=len(images) * time_per_step,
                     true_image=True, salience=False)

        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL, lineterminator = '\n')
        wr.writerow([i, np.mean(jList[-summaryLength:]), np.mean(rList[-summaryLength:]),
                     path + '/frames/image' + str(i) + '.gif',
                     path + '/frames/log' + str(i) + '.csv', 
                     path + '/frames/sal' + str(i) + '.gif'])
        
        myfile.close()
        
    with open(path + '/frames/log' + str(i) + '.csv', 'w') as myfile:
        state_train = (np.zeros([1, h_size]), np.zeros([1, h_size]))
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL, lineterminator = '\n')
        
        #Hard-coded number of actions. Need to work on that!
        #wr.writerow(["ACTION", "REWARD", "A0", "A1", 'A2', 'V'])
        '''
        a_size = 3
        wr.writerow(["ACTION", "REWARD"] +  ["A" + str(act_idx) for act_idx in range(0, a_size)] + ['V'])
        
        test_im = list(zip(bufferArray[:, 0]))
        test_im = np.vstack(test_im)
           
        
        if chls == 2:
            test_s = test_im[:,0:-img_x*img_x]
        else:
            test_s = test_im
          
        a, v = sess.run([mainQN.Advantage, mainQN.Value],
                        feed_dict={mainQN.scalarInput: np.vstack(test_s) / 255.0,
                                   mainQN.trainLength: len(bufferArray), 
                                   mainQN.state_in: state_train,
                                   mainQN.batch_size: 1})
        '''
        #Hard-code number of actions Need to work on that
        #a_comprehension = [a[:, act_idx] for act_idx in range(0,a_size)]
        #a_list = [bufferArray[:, 1] + bufferArray[:, 2]] + a_comprehension + [v[:, 0]]
        #to_write = list(zip(a_list))
        #wr.writerows(to_write)
        #wr.writerows(list(zip(bufferArray[:, 1], bufferArray[:, 2], a[:,0], a[:,1], a[:,2], v[:, 0])))


# This code allows gifs to be saved of the training episode for use in the Control Center.
def make_gif(images, fname, duration=2, true_image=False, salience=False, salIMGS=None):
    import moviepy.editor as mpy

    def make_frame(t):
        try:
            #print('trying to print...')
            x = images[int(len(images) / duration * t)]
            #print(x)
        except:
            x = images[-1]

        if true_image:
            return x.astype(np.uint8)
        else:
            return ((x + 1) / 2 * 255).astype(np.uint8)

    def make_mask(t):
        try:
            x = salIMGS[int(len(salIMGS) / duration * t)]
        except:
            x = salIMGS[-1]
        return x

    clip = mpy.VideoClip(make_frame, duration=duration)
    if salience == True:
        mask = mpy.VideoClip(make_mask, ismask=True, duration=duration)
        clipB = clip.set_mask(mask)
        clipB = clip.set_opacity(0)
        mask = mask.set_opacity(0.1)
        mask.write_gif(fname, fps=len(images) / duration, verbose=False)
    # clipB.write_gif(fname, fps = len(images) / duration,verbose=False)
    else:
        clip.write_gif(fname, fps=len(images) / duration, verbose=False)