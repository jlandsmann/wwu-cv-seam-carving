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