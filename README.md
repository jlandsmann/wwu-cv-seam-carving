# wwu-cv-seam-carving

We tried different networks and tested them on the same dataset.
The first network is based on the paper 
"Seam Carving Detection Using Convolutional Neural Networks".
It will be referenced as "CNN".
The second network we tried is described by the article
"Deep Convolutional Neural Network for Identifying Seam-Carving Forgery"
and will be referenced as "DCNN".

## rgb based

| Network   | Dataset   | Batch Size | Epochs | Learning rate | Avg. Loss | Accuracy |
| --------- | --------- | ---------- | ------ | ------------- | --------- | -------- |
| DCNN-1    | First 512 | 64         | 10     | 1e-3          | 0.673017  | 68.9%    |
| DCNN-2    | First 512 | 16         | 10     | 1e-2          | 0.645639  | 65.2%    |
| DCNN-3    | First 256 | 16         | 5      | 1e-1          | 0.645639  | 69.5%    |
| DCNN-4    | First 256 | 16         | 5      | 1e+2          | 0.617949  | 69.5%    |
| DCNN-4    | First 256 | 16         | 20     | 1e+2          | 0.617949  | 69.5%    |

## grayscale based

| Network   | Dataset   | Batch Size | Epochs | Learning rate | Avg. Loss | Accuracy |
| --------- | --------- | ---------- | ------ | ------------- | --------- | -------- |
| DCNN-5    | First 512 | 16         | 10     | 1e-3          | 0.636411  | 69.3%    |
| DCNN-6    | First 512 | 32         | 10     | 1e-3          | 0.634630  | 69.3%    |
| DCNN-7    | First 512 | 32         | 20     | 1e-3          | 0.625384  | 69.3%    |

## New optimizer
In the previous tests we observed that the accuracy 
and the loss were very stable after a few epochs.
But the accuracy is quite low, so we tested a different optimzer.
Instead of the SGD-Optimizer we then started to use the Adam-optimizer,
which is also used in the paper
"Deep Convolutional Neural Network for Identifying Seam-Carving Forgery".

| Network     | Dataset   | Batch Size | Epochs | LR   | b1, b2     | eps  | Avg. Loss | Accuracy |
| ----------- | --------- | ---------- | ------ | ---- | ---------- | ---- | --------- | -------- |
| DCNN-ADAM-1 | First 512 | 16         | 10     | 1e-3 | 0.9 , 0.99 | 1e-8 | 0.635976  | 65.0%    |
| DCNN-ADAM-1 | First 512 | 16         | 20     | 1e-3 | 0.9 , 0.99 | 1e-8 | 0.665614  | 60.9%    |
| DCNN-ADAM-2 | First 512 | 16         | 7      | 1e-3 | 0.9 , 0.99 | 1e-4 | 0.665614  | 65.2%    |
| DCNN-ADAM-2 | First 512 | 16         | 8      | 1e-3 | 0.9 , 0.99 | 1e-4 | 0.665614  | 40.6%    |
| DCNN-ADAM-2 | First 512 | 16         | 10     | 1e-3 | 0.9 , 0.99 | 1e-4 | 0.665614  | 50.4%    |
| DCNN-ADAM-2 | First 512 | 16         | 20     | 1e-3 | 0.9 , 0.99 | 1e-4 | 0.651364  | 65.2%    |
| DCNN-ADAM-2 | First 512 | 16         | 30     | 1e-3 | 0.9 , 0.99 | 1e-4 | 0.674053  | 61.9%    |
| DCNN-ADAM-3 | All       | 16         | 10     | 1e-3 | 0.9 , 0.99 | 1e-4 | 0.615799  | 67.6%    |
| DCNN-ADAM-3 | All       | 16         | 20     | 1e-3 | 0.9 , 0.99 | 1e-4 | 0.611167  | 67.7%    |

During the first test with the new optimizer (DCNN-ADAM-1)
we observed a way more variation of accuracy and loss.
That's the reason why we decided to keep the new optimizer
altough the accuracy is lower than before. 
We noticed that the accuracy still converges,
so we tested different optimizer and loss functions.
But neither had a noticeable impact.

So we set up the theorie our network 
has too many paramters for too little data.
The fact that we cannot overfit our model
by using the same data for training and testing.

## Smaller network
That the reason why we reduced our network size
to ensure that our paramter count is not too large.
With the network from the paper
"Seam Carving Detection Using Convolutional Neural Networks"
we tried first to provocate overfitting
by training and testing on the same dataset.
After that worked we moved forward
and started new models with separated test- and trainingsdata.

| Network    | Dataset   | Batch Size | Epochs | LR   | b1, b2     | eps  | Avg. Loss | Accuracy |
| ---------- | --------- | ---------- | ------ | ---- | ---------- | ---- | --------- | -------- |
| CNN-ADAM-1 | All       | 8          | 10     | 1e-5 | 0.9 , 0.99 | 1e-6 | 0.599068  | 70.1%    |


We introduced a new optimizer SGD with less hyper parameters
so the optimization process of the hyper parameters is less complicated.

| Network   | Dataset   | Batch Size | Epochs | LR   | momentum  | Avg. Loss | Accuracy |
| --------- | --------- | ---------- | ------ | ---- | --------- | --------- | -------- |
| CNN-SGD-1 | All       | 8          | 10     | 1e-5 | 0         | 0.657398  | 60.4%    |
| CNN-SGD-2 | All       | 8          | 10     | 1e-7 | 9e-1      | 0.682758  | 57.2%    |
| CNN-SGD-3 | All       | 8          | 10     | 1e-3 | 9e-1      | 0.710225  | 54.8%    |
| CNN-SGD-4 | All       | 4          | 10     | 1e-3 | 99e-2     | 0.689581  | 61.3%    |

Update number of dimensions in convolutional layers:
From 1 -> 8 -> 5 -> 1
To 1 -> 3 -> 2 -> 1     => overfitting, only labelling carved

To 1 -> 4 -> 16 -> 1
Epochs=10 LR=1e-3 MOM=99e-2
Accuracy: 58.8%, Avg loss: 0.702854 