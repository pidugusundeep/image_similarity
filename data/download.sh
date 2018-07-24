mkdir -p holiday-photos/image
cd holiday-photos

wget ftp://ftp.inrialpes.fr/pub/lear/douze/data/jpg1.tar.gz
wget ftp://ftp.inrialpes.fr/pub/lear/douze/data/jpg2.tar.gz

cd image

tar xvzf ../jpg1.tar.gz
tar xvzf ../jpg2.tar.gz
