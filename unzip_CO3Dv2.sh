DATA_ROOT=/data/wanghaoxuan/CO3Dv2_single_seq/
for z in /data/wanghaoxuan/CO3Dv2_single_seq/*.zip; do
  echo "→ $z"
  unzip -q -n "$z" -d "$DATA_ROOT"
done
echo "Done unzipping."