"""Create an Image Classification Web App using PyTorch and Streamlit."""
# import libraries

import time
from PIL import Image
from torchvision import models, transforms
import torch
import streamlit as st

# set title of app
st.title("Simple Image Classification Application")
st.write("")

# enable users to upload images for the model to make predictions
file_up = st.file_uploader("Upload an image", type = "jpg")
st.sidebar.markdown("# Pleace Choose A Model To Classify An Image")
model_dict = {'VGG16-bn':models.vgg16_bn,'VGG19-bn':models.vgg19_bn,
              'AlexNet':models.alexnet,'ResNet50':models.resnet50,
              'ResNet101':models.resnet101,'ResNet152':models.resnet152,
              'DenseNet121':models.densenet121,'DenseNet169':models.densenet169,'DenseNet201':models.densenet201,
              'GoogleNet':models.googlenet,'Inception V3':models.inception_v3,
              'ShuffleNet V2 x1.0':models.shufflenet_v2_x1_0,
              'SqueezeNet 1.0':models.squeezenet1_0,'SqueezeNet 1.1':models.squeezenet1_1,
              'MobileNet V2':models.mobilenet.mobilenet_v2,
              'MobileNet V3 Large':models.mobilenet.mobilenet_v3_large,'MobileNet V3 small':models.mobilenet.mobilenet_v3_small}

model_name = st.sidebar.selectbox("Models In Pytorch",
                                      list(model_dict.keys()))

print('choose ',model_name)

def predict(image,model_name):
    """Return top 5 predictions ranked by highest probability.

    Parameters
    ----------
    :param image: uploaded image
    :type image: jpg
    :rtype: list
    :return: top 5 predictions ranked by highest probability
    """
    # create a model
    # resnet = models.resnet101
    pre_model = model_dict[model_name](pretrained=True)
    device = 'cuda' if torch.cuda.is_available else  'cpu'

    device = 'cpu'
    pre_model = pre_model.to(device)
    # transform the input image through resizing, normalization
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
            )])

    # load the image, pre-process it, and make predictions
    img = Image.open(image)
    batch_t = torch.unsqueeze(transform(img), 0)
    pre_model.eval()
    batch_t = batch_t.to(device)
    out = pre_model(batch_t)
    # model_vis = make_dot(out, params=dict(list(pre_model.named_parameters()) + [('img', batch_t)]))
    # model_vis.format = "png"
    # # 指定文件生成的文件夹
    # model_vis.directory = "./"
    # # 生成文件
    # model_vis.view(False)
    
    # hl_graph = hl.build_graph(pre_model, torch.zeros([1, 3, 224, 224]).to(device))
    # hl_graph.theme = hl.graph.THEMES["blue"].copy()  # Two options: basic and blue
    # hl_graph.save(path='./demoModel.png', format='png')  # 保存网络模型图，可以设置 png 和 pdf 等
    with open('imagenet_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]

    # return the top 5 predictions ranked by highest probabilities
    prob = torch.nn.functional.softmax(out, dim = 1)[0] * 100
    _, indices = torch.sort(out, descending = True)
    return [(classes[idx], prob[idx].item()) for idx in indices[0][:5]]


if file_up is not None:
    # display image that user uploaded
    image = Image.open(file_up)
    st.write("We use ",model_name,"To Predict")
    st.image(image, caption = 'Uploaded Image.', use_column_width = True)
    st.write("")
    st.write("Just a second ...")
    start = time.time()
    labels = predict(file_up,model_name)    
    end = time.time()
    
    # print out the top 5 prediction labels with scores
    for i in labels:
        st.write("Prediction (index, name)", i[0], ",   Score: ", i[1])
    st.write('It takes ',end - start,' s')
    
    # image = Image.open('demoModel.png')
    # st.image(image,caption=model_name,use_column_width=True)
