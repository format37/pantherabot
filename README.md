# panthera bot
Conversational telegram bot docker server  
![Structure](assets/structure_v1.png)  
* [Telebot server](https://github.com/format37/telegram_bot)  
* [Panthera bot](https://github.com/format37/pantherabot)  
* [LLM service](https://github.com/format37/openai_proxy)
# root documents and photos mounting
":" are not suported in mouting therefore we need to remove user_id from mounting procedure:
```
sudo mount --bind "/user_id:token" "/mnt/token"
```
To provide mounting after reboot:
```
echo '"/user_id:TOKEN" /mnt/TOKEN none bind 0 0' | sudo tee -a /etc/fstab
```