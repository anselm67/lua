#include <luaT.h>
#include <TH/TH.h>

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>

#ifndef TRUE
#  define TRUE 1
#endif

#ifndef FALSE
#  define FALSE 0
#endif

static const char *TYPE_NAME = "LuaVideo.decoder";

typedef struct lua_video {
  char *filename;
  AVFormatContext *container;
  AVCodecContext *dec_ctx;
  AVCodec *codec;
  AVFrame *frame;
  AVFrame *rgb;
  struct SwsContext *sws_ctx;
  int video_index;
  int failed;
} lua_video_t;

static int  video_open(const char *filename);
static void video_close(lua_video_t *);
static int  video_frame(lua_video_t *);
static lua_video_t *check_video(lua_State *L, int check_failed);

static int video_init(const char *filename, lua_video_t *v) {
  v->failed = TRUE;

  if (! (v->filename = strdup(filename)) ) {
    video_close(v);
    return FALSE;
  }
  
  // Opens and grabs the stream info.
  if (avformat_open_input(&(v->container), filename, NULL, NULL) != 0 ||
      avformat_find_stream_info(v->container, NULL) < 0) {
    video_close(v);
    return FALSE;
  }

  // Looks up the video stream within the container.
  v->dec_ctx = v->container->streams[v->video_index]->codec;
  v->video_index = av_find_best_stream(v->container, AVMEDIA_TYPE_VIDEO, -1, -1,
				       &(v->codec), 0);
  if (v->video_index < 0) {
    video_close(v);
    return FALSE;
  }
  
  if (avcodec_open2(v->dec_ctx, v->codec, NULL) < 0) {
    video_close(v);
    return FALSE;
  }

  if (! (v->frame = av_frame_alloc()) ||
      ! (v->rgb = av_frame_alloc()) ) {
    video_close(v);
    return FALSE;
  }

  v->rgb->format = AV_PIX_FMT_RGB24;
  v->rgb->width  = v->dec_ctx->width;
  v->rgb->height = v->dec_ctx->height;
  if (av_frame_get_buffer(v->rgb, 32) < 0) {
    video_close(v);
    return FALSE;
  }
  
  v->sws_ctx = sws_getContext(
    v->dec_ctx->width, v->dec_ctx->height,
    v->dec_ctx->pix_fmt,
    v->dec_ctx->width, v->dec_ctx->height,
    AV_PIX_FMT_RGB24,
    SWS_BILINEAR,
    NULL, NULL, NULL);
    
  v->failed = FALSE;
  return TRUE;
}

static void video_close(lua_video_t *v) {
  if (v->filename) {
    free(v->filename);
    v->filename = NULL;
  }
  if (v->dec_ctx) {
    avcodec_close(v->dec_ctx);
    v->dec_ctx = NULL;
  }
  if (v->container) {
    avformat_close_input(&(v->container));
    v->container = NULL;
  }
  if (v->frame) {
    av_frame_free(&(v->frame));
    v->frame = NULL;
  }
  if (v->sws_ctx) {
    sws_freeContext(v->sws_ctx);
    v->sws_ctx = NULL;
  }
  v->failed = TRUE;
}

static lua_video_t *check_video(lua_State *L, int check_failed) {
  lua_video_t *v = (lua_video_t *) luaL_checkudata(L, 1, TYPE_NAME);
  luaL_argcheck(L, v != NULL, 1, "'video' expected.");
  if (check_failed) {
    luaL_argcheck(L, ! v->failed, 1, "valid 'video' expected.");
  }
  return v;
}

static int video_frame(lua_video_t *v) {
  AVPacket packet;
  int ret = FALSE;
  
  while(av_read_frame(v->container, &packet) >= 0) {
    if (packet.stream_index == v->video_index) {
      int done = 0;
      if (avcodec_decode_video2(v->dec_ctx, v->frame, &done, &packet) < 0) {
	v->failed = TRUE;
	break;
      }
      if ((ret = done)) {
	sws_scale(v->sws_ctx,
		  (const uint8_t * const *) v->frame->data, v->frame->linesize,
		  0, v->dec_ctx->height,
		  v->rgb->data, v->rgb->linesize);
	break;
      }
    }
  }
  av_packet_unref(&packet);

  return ret;
}

static int lua_video_open(lua_State *L) {
  const char *filename = luaL_checkstring(L, 1);

  lua_video_t *video = (lua_video_t *) lua_newuserdata(L, sizeof(lua_video_t));
  memset(video, 0, sizeof(lua_video_t));

  if (video_init(filename, video)) {
    luaL_getmetatable(L, TYPE_NAME);
    lua_setmetatable(L, -2);
  } else {
    luaL_error(L, "cannot open video %s", filename);
  }
  return 1;
}

static int lua_video_close(lua_State *L) {
  lua_video_t *video = check_video(L, FALSE);
  if (video) {
    video_close(video);
  }
  lua_pushboolean(L, TRUE);
  return 1;
}

static int lua_video_isvalid(lua_State *L) {
  lua_video_t *video = check_video(L, FALSE);
  lua_pushboolean(L, ! video->failed);
  return 1;
}

static int lua_video_size(lua_State *L) {
  lua_video_t *video = check_video(L, TRUE);

  lua_pushinteger(L, video->dec_ctx->width);
  lua_pushinteger(L, video->dec_ctx->height);
  return 2;
}

static int lua_video_grab(lua_State *L) {
  lua_video_t  *video = check_video(L, TRUE);
  luaL_checkudata(L, 2, "torch.ByteTensor");
  THByteTensor *out = luaT_toudata(L, 2, luaT_typenameid(L, "torch.ByteTensor"));
  
  if (out->nDimension != 3 || out->size[0] != 3 ||
      out->size[1] != video->dec_ctx->height ||
      out->size[2] != video->dec_ctx->width) {
    luaL_error(L, "Expcting a byte tensor with size x3x%dx%d",
	       video->dec_ctx->height, video->dec_ctx->width);
  }

  if (video_frame(video)) {
    uint8_t *dst = THByteTensor_data(out);
    AVFrame *rgb = video->rgb;

    uint8_t *r = dst, *g = r + out->stride[0], *b = g + out->stride[0];
    
    for (int y = 0; y < rgb->height; y++) {
      uint8_t *src = rgb->data[0] + y * rgb->linesize[0];
      for (int x = 0; x < rgb->width; x++) {
	*r++ = *src++;
	*g++ = *src++;
	*b++ = *src++;
      }
    }
    
    lua_pushboolean(L, TRUE);
  } else {
    lua_pushboolean(L, FALSE);
  }
  return 1;
}

static int lua_video_fps(lua_State *L) {
  lua_video_t *video = check_video(L, TRUE);
  AVStream *stream = video->container->streams[video->video_index];

  double fps = stream->r_frame_rate.num / ((double) stream->r_frame_rate.den);

  lua_pushnumber(L, fps);
  return 1;
}

static int lua_video_length(lua_State *L) {
  lua_video_t *video = check_video(L, TRUE);
  AVStream *stream = video->container->streams[video->video_index];

  lua_pushnumber(L, stream->nb_frames);
  return 1;
}

static const struct luaL_reg videolib_methods[] = {
  {"close", lua_video_close},
  {"isvalid", lua_video_isvalid},
  {"size", lua_video_size},
  {"grab", lua_video_grab},
  {"fps", lua_video_fps},
  {"length", lua_video_length},
  {"__gc", lua_video_close},
  {NULL, NULL}
};

static const struct luaL_reg videolib_funcs[] = {
  {"open", lua_video_open},
  {NULL, NULL}
};

DLL_EXPORT int luaopen_libvideo(lua_State *L) {
  av_register_all();

  luaL_newmetatable(L, TYPE_NAME);
  lua_pushliteral(L, "__index");
  lua_pushvalue(L, -2);
  lua_rawset(L, -3);

  luaL_openlib(L, NULL, videolib_methods, 0);
  
  luaL_openlib(L, "video", videolib_funcs, 0);
  return 1;
}
