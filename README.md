# Introduction
Discord bot can correct your obvious errors in voice content, which can improve your english spaeking ability, especially for IETLS speaking test.

# How to use

install packages
```
pip install requirements.txt
```

Run
```
python3 voiceCorrection.py
```

As a result, This program will direct discord bot to send message to you, messages like:
> I want to talk about the balcony which is connected ~~with~~to my living room. There are plenty of spaces ~~i~~on my balcony, and I can do plenty of activities ion my balcony. It can help me to recharge my battery.

illustrate: it corrects 
+  "in my balcony" to "on my balcony"
+  "is connected with" to "is connected to"

# Problem
This project can be used by personal ustage, if you want to publize to public, there are several problems need to be solve
+ some content cannot be correct thought LLM, because my computer doesn't have enough great performance to run LLM for multi threads. better compute can optimize this problem.
+ The order of message is not total same with voice, because I do not reorder it after I receive the result from multi thread process. I havn't learned this technique.
+ the result of text-to-speech are not always good, large model will be better, but base verison is enough to use if you speack with the standard of ielts speaking test.