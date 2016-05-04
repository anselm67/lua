
Simple interface to libavcodec for lua. 

A sample is wth 1,000 words:

> require 'video'
> v = video.open('foo.mp4')   -- Opens a video stream.
> width, height = v:size()    -- Retreives its dimension.
> fps = v:fps()               -- What's the fps?
> len = v:length()            -- How many frames will I get?
> tensor = torch.ByteTensor(3, height, width)  -- You can only read into a byte tensor with 3 channels.
> while v:grab(tensor) do <something> end
> v:close()

That's it! Why did I write this?
- There is an old libavcodec extension, but it has a bunch of static that prevents from opening multiple streams within the same session,
- Another one (ffmpeg?) uses ffmpeg as a command line tool to extract frames to dsik.

So this extension, while it suffers from many other drawbacks, doesn't suffer from these.

Enjoy!

