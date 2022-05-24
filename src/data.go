package main

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"io"
	"os"

	"github.com/nfnt/resize"
)

var type_map map[uint32]string = map[uint32]string{
	0x8: "unsigned byte",
	0x9: "signed byte",
	0xB: "short",
	0xC: "int",
	0xD: "float",
	0xE: "double",
}

type labels struct {
	magic_num      uint32
	data_type      string
	dimensionality uint
	n              uint32
	labels         []byte
}

type images struct {
	magic_num      uint32
	data_type      string
	dimensionality uint
	n              uint32
	img_rows       uint32
	img_cols       uint32
	images         [][][]float64
}

func read_labels(fpath string) (labels, error) {
	fd, err := os.Open(fpath)
	if err != nil {
		return labels{}, err
	}
	defer fd.Close()

	i_buff, b_buff := make([]byte, 8), make([]byte, 1)
	if _, err := fd.Read(i_buff); err != nil {
		return labels{}, err
	}

	mnum := to_big_endian(i_buff[:4])

	lbs := labels{
		magic_num: mnum, n: to_big_endian(i_buff[4:]),
		data_type: type_map[mnum&(0xFF>>8)], dimensionality: uint(mnum & 0xFF),
	}

	lbs.labels = make([]byte, lbs.n)

	i := 0
	for {
		if _, err := fd.Read(b_buff); err != nil {
			if err == io.EOF {
				return lbs, nil
			}
			return labels{}, err
		}
		lbs.labels[i] = b_buff[0]
		i++
	}
}

func to_big_endian(data []byte) uint32 {
	return uint32(data[0])<<24 | uint32(data[1])<<16 | uint32(data[2])<<8 | uint32(data[3])
}

func read_imgs(fpath string) (images, error) {
	fd, err := os.Open(fpath)
	if err != nil {
		return images{}, err
	}
	defer fd.Close()

	i_buff := make([]byte, 16)
	if _, err := fd.Read(i_buff); err != nil {
		return images{}, err
	}

	mnum := to_big_endian(i_buff[:4])

	imgs := images{
		magic_num: mnum, n: to_big_endian(i_buff[4:8]),
		img_rows: to_big_endian(i_buff[8:12]), img_cols: to_big_endian(i_buff[12:]),
		data_type: type_map[mnum&(0xFF>>8)], dimensionality: uint(mnum & 0xFF),
	}

	imgs.images = make([][][]float64, imgs.n)
	for i := range imgs.images {
		imgs.images[i] = make([][]float64, imgs.img_rows)
		for j := range imgs.images[i] {
			imgs.images[i][j] = make([]float64, imgs.img_cols)
		}
	}

	i := 0
	b_buff := make([]byte, imgs.img_rows*imgs.img_cols)
	for {
		if _, err := fd.Read(b_buff); err != nil {
			if err == io.EOF {
				return imgs, nil
			}
			return images{}, err
		}
		for j := 0; j < int(imgs.img_rows); j++ {
			for k := 0; k < int(imgs.img_cols); k++ {
				imgs.images[i][j][k] = float64(b_buff[j*int(imgs.img_cols)+k]) / 255
			}
		}
		i++
	}
}

func dump_image(imgs images, index int, fname string) error {
	img := image.NewRGBA(image.Rect(0, 0, int(imgs.img_cols), int(imgs.img_rows)))

	for i, row := range imgs.images[index] {
		for j, px := range row {
			img.SetRGBA(j, i, color.RGBA{255 - uint8(px*255), 255 - uint8(px*255), 255 - uint8(px*255), 255})
		}
	}

	rsz_img := resize.Resize(100, 100, img, resize.Lanczos3)

	fd, err := os.Create(fname)
	if err != nil {
		return fmt.Errorf("couldn't create the image: %v", err)
	}
	defer fd.Close()

	return png.Encode(fd, rsz_img)
}
