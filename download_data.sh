fileId=1ep9H6-HvhB4utJRLVcLzieWNUSG3P_uF
fileName=test.csv
curl -sc ./cookie "https://drive.google.com/uc?export=download&id=${fileId}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' ./cookie)"
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${fileId}" -o ${fileName}
rm ./cookie
