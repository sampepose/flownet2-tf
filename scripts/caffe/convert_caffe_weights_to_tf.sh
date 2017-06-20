#!/bin/sh
die () {
    echo >&2 "$@"
    exit 1
}

SCRIPTPATH=$( cd $(dirname $0) ; pwd -P )
[ "$#" -eq 1 ] || die "1 argument required, $# provided"
case $1 in
    "C" )
    tmp="${SCRIPTPATH}/flownet-C.npy"
    ;;
    "S" )
    tmp="${SCRIPTPATH}/flownet-S.npy"
    ;;
    "2" )
    tmp="${SCRIPTPATH}/flownet-2.npy"
    ;;
    * )
    die "argument must be C or S or 2"
    ;;
esac

python ./convert_caffe_weights_to_npy.py --out $tmp --arch $1
python ./convert_npy_weights_to_tf.py --input $tmp
rm $tmp
