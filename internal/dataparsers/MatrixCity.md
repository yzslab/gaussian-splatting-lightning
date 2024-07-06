* Structure
```bash
├── small_city/aerial
    ├── block_1
        ├── depth  # provided in the tar files here: https://huggingface.co/datasets/BoDai/MatrixCity/tree/main/small_city_depth
            ├── 0000.exr
            ├── 0001.exr
            ...
        ├── rgb  # provided in the tar files here: https://huggingface.co/datasets/BoDai/MatrixCity/tree/main/small_city
            ├── 0000.png
            ├── 0001.png
            ...
        ├── transforms.json  # Poses of images after removing the images that look outside the map boundary, which are used for training and testing
        ├── transforms_origin.json  # Poses of all original collected images
    ├── block_1_test
        ...  # same as `block_1` above
    ├── block_2
        ...  # same as `block_1` above
    ...
├── small_city/street
    ...  # same as `small_city/aerial` above
```

Example contents of `transforms.json` and `transforms_origin.json`:
```json
{
    "camera_angle_x": 0.7853981852531433,
    "frames": [
        {
            "frame_index": 1,
            "rot_mat": [
                [
                    -4.3711387287537207e-10,
                    0.0070710680447518826,
                    -0.007071067579090595,
                    -1000.0
                ],
                [
                    -0.009999999776482582,
                    -5.844237316310341e-10,
                    3.374863583038845e-11,
                    -38.0
                ],
                [
                    -3.89386078936127e-10,
                    0.007071067579090595,
                    0.0070710680447518826,
                    150.0
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0
                ]
            ],
            "euler": [
                0.7853981256484985,
                3.893860878179112e-08,
                -1.5707963705062866
            ]
        },
        // other frames
   ]
}
```

* DataParser params example

```bash 
python main.py fit \
    --data.path data/MatrixCity/small_city/aerial \
    --data.parser MatrixCity \
    --data.parser.train '["block_1/transforms.json", "block_2/transforms.json"]' \
    --data.parser.test '["block_1_test/transforms.json", "block_2_test/transforms.json"]' \
    ...
```

The `--data.parser.train` and `--data.parser.test` specify the json files of the blocks you want to use.