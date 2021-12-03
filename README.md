# Tomato-Leaf-Disease-Detection-DHL

##### Profile Visits :
![Visitors](https://visitor-badge.glitch.me/badge?page_id=ronylpatil.&left_color=lightgrey&right_color=red&left_text=visitors)

<p align="center">
  <img class="center" src ="https://www.treehugger.com/thmb/mjY1JW6RxThRl_Yvv4XaGaBPPTY=/768x0/filters:no_upscale():max_bytes(150000):strip_icc():format(webp)/GettyImages-175396800-a15ecff03062438894935715d9550479.jpg" alt="Drawing" style="width: 1350px; height: 600px">
</p>

<b>Description : </b>
Here I am combining best of the both worlds, one is traditional Machine Learning and, another is Deep Learning to create a hybrid network which classifies Tomato Leaf Diseases.
From Deep Learning I took VGG16 Pre-trained Network for extracting useful features and, from Machine Learning I tried out both XGBOOST and, Random Forest Ensembel Techniques to classify the images. We can call it as Hybrid AI or Hybrid Learning. I call it "Deep Hybrid Learning", because here we are using fusion of Machine Learning and Deep Learning to create Hybrid Model. 
               
Actually this hybrid model works amazing well specially when we have limited training dataset.
               If we have lot's of data then deep learning works well as compare to traditional machine learning. But in case of very limited data
               ten's of images rather than thousands of images then traditional machine learning works great. Now here we have taken best 
               of both worlds I mean from deep learning world instead of doing deep learning let's only extract features using VGG16 Pre-trained Network and then take that responce or features to train random forest and, xgboost for classification. 

In this project I am classifying 5 categories of tomato plant leaf disease(including healthy leaves) and took total 2500 images(500 of each category) and, for testing I took 1000 images(200 of each category) and you will don't believe that Random Forest gave 83% of accuracy(without model tunning) and, XGBOOST gave 89% of accuracy(without model tunning) which is far better and, the hybrid model trained on 2500 images in just 7 minutes.   
               
<b>Dataset Source : https://www.kaggle.com/arjuntejaswi/plant-village</b>

<b>Model Performace :</b>
i. Random Forest Performance 
<p align="center">
  <img class="center" src ="/sample/random forest cm.png" alt="Drawing" style="width: 500px; height: 400px">
</p>

<p align="center">
  <img class="center" src ="/sample/random forest cr.png" alt="Drawing" style="width: 500px; height: 230px">
</p>

ii. XGBOOST Performance 
<p align="center">
  <img class="center" src ="/sample/xgb cm.png" alt="Drawing" style="width: 500px; height: 400px">
</p>

<p align="center">
  <img class="center" src ="/sample/xgb cr.png" alt="Drawing" style="width: 500px; height: 230px">
</p>

<p align="center">
  <a href="https://www.linkedin.com/in/ronylpatil/">Made with ❤ by ronil</a>
</p>
