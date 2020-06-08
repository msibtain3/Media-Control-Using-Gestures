import vlc

Instance = vlc.Instance()
player = Instance.media_player_new()
Media = Instance.media_new("C:/Users/MSIBT/Downloads/Video/Tom & Jerry - Top 10 Classic Chase Scenes - Classic Cartoon - WB Kids - YouTube.mkv")    
player.set_media(Media)