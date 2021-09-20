import vlc

Instance = vlc.Instance()
player = Instance.media_player_new()
Media = Instance.media_new("path to the video file")    
player.set_media(Media)
