# stepin

Step inside a photo, in your terminal, and run shell commands on what you find there.

![demo](docs/demo.gif)

```
$ stepin livingroom.jpg
livingroom> ls
bookcase/  chair/  lamp  lamp2/  painting  rug  sofa/  table  vase
livingroom> cd sofa
livingroom/sofa> cd ..
livingroom> rm table
removed table
painted in what might be behind it
livingroom> find red mug
```

`cd` flies you there. `rm` blows the thing to dust and shows what might have been behind it. `rm sofa` refuses, because
it has cushions in it: that takes `rm -r`.

## Run

Needs Python 3.11 to 3.13 and Linux (developed on WSL2 with a 12 GB NVIDIA card; it uses about 6 GB). Without a GPU it
should work but looking at a photo will take minutes.

```
uv sync
uv run stepin photo.jpg
```

The first run downloads about 5 GB of models. A new photo takes about half a minute to look at; after that it opens in
a couple of seconds. Everything runs on your machine: photos never leave it.

## Controls

| | |
| --- | --- |
| `W` `A` `S` `D`, `R` `F` | move, up and down |
| arrows, or drag | look around |
| wheel | forward, back |
| click | what is this? (while typing a command: insert its name) |
| `Enter` | open the shell; on an empty line, back to flying |
| `M` / `T` / `L` | drawing mode / style / name tags |
| `Home` | back to where the photo was taken |
| `Q` | quit |

Type `help` in the shell for the commands. Tab completes, up and down recall.

## How it works

1. **[MoGe-2](https://github.com/microsoft/MoGe)** estimates where every pixel is, in metres, and the camera's angle of view.
2. **[Florence-2](https://huggingface.co/florence-community/Florence-2-large)** and
   **[Grounding DINO](https://huggingface.co/IDEA-Research/grounding-dino-base)** name what is there.
   `find` asks Grounding DINO for whatever you type.
3. **[Segment Anything 2.1](https://huggingface.co/facebook/sam2.1-hiera-large)** cuts each thing out. Things inside the
   outline of a bigger thing, at the same depth, become its children.
4. **[LaMa](https://github.com/advimman/lama)** paints over them. MoGe-2 measures the repainted photo, and that depth is
   fitted to the real surroundings. This is the hidden layer you see when you step sideways.
5. The result is about a million points. `render.py` draws them all in about 10 ms per frame on a laptop CPU: plain
   Python compiled by Numba, with a z-buffer split over the cores.
6. `cells.py` turns each frame into characters and sends only the cells that changed.

The AI runs in its own process, so the picture keeps moving while it thinks.

## Limits

- A photo shows one side of things. What is behind them is a guess, and some gaps stay empty.
- Sizes come from one picture: good for "about 1.5 m", not for ordering a shelf. Only the visible part is measured.
- Edits last until you quit.
- Shadows stay behind when you remove what cast them.

## Files

Scenes, the LaMa weights and the log are in `~/.cache/stepin`, the other models in `~/.cache/huggingface`.
`stepin --rebuild photo.jpg` looks at a photo again.

## Test

```
uv run pytest
```

No models or GPU needed.

## Related

[tortuise](https://github.com/buildoak/tortuise) and [tsplat](https://github.com/darshanmakwana412/tsplat) view Gaussian
splats in the terminal.
