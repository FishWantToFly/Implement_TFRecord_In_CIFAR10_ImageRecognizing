
# coding: utf-8

# In[1]:


import os
import tensorflow as tf
from PIL import Image

classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
cwd = os.getcwd()
print(cwd)

train_data_path = cwd + '/train' 
print(train_data_path)



# In[2]:


def create_record():
    '''
    此处我加载的数据目录如下：
    0 -- img1.jpg
         img2.jpg
         img3.jpg
         ...
    1 -- img1.jpg
         img2.jpg
         ...
    2 -- ...
    ...
    '''
    writer = tf.python_io.TFRecordWriter("train.tfrecords")
    for index, name in enumerate(classes):
        class_path = train_data_path + "/" + name 
        #print(class_path)
        for img_name in os.listdir(class_path):
            img_path = class_path + '/' + img_name
            
            img = Image.open(img_path)
            img = img.resize((227, 227))
            img_raw = img.tobytes() #将图片转化为原生bytes
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
            writer.write(example.SerializeToString())
        print('finish part ', index, ' TFReocrd writing')
    writer.close()


# In[ ]:


def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)

    img = tf.reshape(img, [227, 227, 3])
 
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)

    return img, label


# In[ ]:


if __name__ == '__main__':
    # create TFRecord from /train
    # create_record()
    img, label = read_and_decode("train.tfrecords")
    
    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size=30, capacity=2000,
                                                    min_after_dequeue=1000)
    #初始化所有的op
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
	#启动队列
        
        threads = tf.train.start_queue_runners(sess=sess)
       
        for i in range(3):
          
            
            print(img_batch.shape)
            print(label_batch.shape)
             
            val, l = sess.run([img_batch, label_batch])
           
            #l = to_categorical(l, 12)
            print(val.shape, l)


# In[ ]:




