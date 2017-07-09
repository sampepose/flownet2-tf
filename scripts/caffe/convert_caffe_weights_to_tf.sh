#!/bin/sh
die () {
    echo >&2 "$@"
    exit 1
}

SCRIPTPATH=$( cd $(dirname $0) ; pwd -P )
[ "$#" -eq 1 ] || die "1 argument required, $# provided"
case $1 in
    "C" )
    mkdir "${SCRIPTPATH}/FlowNetC/"
    tmp="${SCRIPTPATH}/FlowNetC/flownet-C.npy"
    ;;
    "S" )
    mkdir "${SCRIPTPATH}/FlowNetS/"
    tmp="${SCRIPTPATH}/FlowNetS/flownet-S.npy"
    ;;
    "CS" )
    mkdir "${SCRIPTPATH}/FlowNetCS/"
    tmp="${SCRIPTPATH}/FlowNetCS/flownet-CS.npy"
    ;;
    "CSS" )
    mkdir "${SCRIPTPATH}/FlowNetCSS/"
    tmp="${SCRIPTPATH}/FlowNetCSS/flownet-CSS.npy"
    ;;
    "CSS-ft-sd" )
    mkdir "${SCRIPTPATH}/FlowNetCSS-ft-sd/"
    tmp="${SCRIPTPATH}/FlowNetCSS-ft-sd/flownet-CSS-ft-sd.npy"
    ;;
    "SD" )
    mkdir "${SCRIPTPATH}/FlowNetSD/"
    tmp="${SCRIPTPATH}/FlowNetSD/flownet-SD.npy"
    ;;
    "2" )
    mkdir "${SCRIPTPATH}/FlowNet2/"
    tmp="${SCRIPTPATH}/FlowNet2/flownet-2.npy"
    ;;
    * )
    die "argument must be C, S, CS, CSS, CSS-ft-sd, SD or 2"
    ;;
esac

python ./convert_caffe_weights_to_npy.py --out $tmp --arch $1
python ./convert_npy_weights_to_tf.py --input $tmp
rm $tmp
