# Messenger Chatbot

Modified chatbot following the Pytorch chatbot tutorial that can generate responses based off of FB Messenger chat feeds.

## How to Use
  - Obtain a `message.json` for desired conversation through Facebook data download
  - Modify `config.json`
    - Change `speaker` to person whose responses to emulate
	- Change `convo_path` to path of `message.json` relative to this directory
	- Change `checkpoint_path` to path of desired checkpoint folder
  - `python train.py`
  - `python test.py`

## Acknowledgements
  - Code written following Matthew Inkawhich's [chatbot tutorial](https://pytorch.org/tutorials/beginner/chatbot_tutorial.html)
  - Adit Deshpande's [blog post](https://adeshpande3.github.io/How-I-Used-Deep-Learning-to-Train-a-Chatbot-to-Talk-Like-Me) used for reference