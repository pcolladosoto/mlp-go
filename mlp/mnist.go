package mlp

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

type Labels struct {
	MagicNum       uint32
	DataType       string
	Dimensionality uint
	N              uint32
	Labels         []byte
}

type Images struct {
	MagicNum       uint32
	DataType       string
	Dimensionality uint
	N              uint32
	ImgRows        uint32
	ImgCols        uint32
	Images         [][][]float64
}

func ReadLabels(fpath string) (Labels, error) {
	fd, err := os.Open(fpath)
	if err != nil {
		return Labels{}, err
	}
	defer fd.Close()

	i_buff, b_buff := make([]byte, 8), make([]byte, 1)
	if _, err := fd.Read(i_buff); err != nil {
		return Labels{}, err
	}

	mnum := toBigEndian(i_buff[:4])

	lbs := Labels{
		MagicNum: mnum, N: toBigEndian(i_buff[4:]),
		DataType: type_map[mnum&(0xFF>>8)], Dimensionality: uint(mnum & 0xFF),
	}

	lbs.Labels = make([]byte, lbs.N)

	i := 0
	for {
		if _, err := fd.Read(b_buff); err != nil {
			if err == io.EOF {
				return lbs, nil
			}
			return Labels{}, err
		}
		lbs.Labels[i] = b_buff[0]
		i++
	}
}

func toBigEndian(data []byte) uint32 {
	return uint32(data[0])<<24 | uint32(data[1])<<16 | uint32(data[2])<<8 | uint32(data[3])
}

func ReadImgs(fpath string) (Images, error) {
	fd, err := os.Open(fpath)
	if err != nil {
		return Images{}, err
	}
	defer fd.Close()

	i_buff := make([]byte, 16)
	if _, err := fd.Read(i_buff); err != nil {
		return Images{}, err
	}

	mnum := toBigEndian(i_buff[:4])

	imgs := Images{
		MagicNum: mnum, N: toBigEndian(i_buff[4:8]),
		ImgRows: toBigEndian(i_buff[8:12]), ImgCols: toBigEndian(i_buff[12:]),
		DataType: type_map[mnum&(0xFF>>8)], Dimensionality: uint(mnum & 0xFF),
	}

	imgs.Images = make([][][]float64, imgs.N)
	for i := range imgs.Images {
		imgs.Images[i] = make([][]float64, imgs.ImgRows)
		for j := range imgs.Images[i] {
			imgs.Images[i][j] = make([]float64, imgs.ImgCols)
		}
	}

	i := 0
	b_buff := make([]byte, imgs.ImgRows*imgs.ImgCols)
	for {
		if _, err := fd.Read(b_buff); err != nil {
			if err == io.EOF {
				return imgs, nil
			}
			return Images{}, err
		}
		for j := 0; j < int(imgs.ImgRows); j++ {
			for k := 0; k < int(imgs.ImgCols); k++ {
				imgs.Images[i][j][k] = float64(b_buff[j*int(imgs.ImgCols)+k]) / 255
			}
		}
		i++
	}
}

func DumpImage(imgs Images, index int, fname string) error {
	img := image.NewRGBA(image.Rect(0, 0, int(imgs.ImgCols), int(imgs.ImgRows)))

	for i, row := range imgs.Images[index] {
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
