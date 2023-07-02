fn multiplyTensors(gId : vec3u, t : u32){
  let curr_m = u32(ping.entries[u32(offset.tensor[t].rows)]);
  let curr_n = u32(ping.entries[u32(offset.tensor[t].cols)]);
  let curr_DataFirstIndex = u32(offset.tensor[t].data);
  let curr_ParentLeft = u32(ping.entries[u32(offset.tensor[t].parents)]);
  let curr_ParentRight = u32(ping.entries[u32(offset.tensor[t].parents) + 1u]);

  // make sure the order ofthe matrices is correct
  let left_is_left = u32(ping.entries[u32(offset.tensor[curr_ParentLeft].isRightMultiplicator)]);
  let right_is_right = u32(ping.entries[u32(offset.tensor[curr_ParentRight].isRightMultiplicator)]);
  if(left_is_left != u32(0) || right_is_right != u32(1)){
    return;
  }
  
  let parentLeft_DataFirstIndex = u32(offset.tensor[curr_ParentLeft].data);
  let parentRight_DataFirstIndex = u32(offset.tensor[curr_ParentRight].data);
  let parentLeft_m = u32(ping.entries[u32(offset.tensor[curr_ParentLeft].rows)]);
  let parentLeft_n = u32(ping.entries[u32(offset.tensor[curr_ParentLeft].cols)]);
  let parentRight_n = u32(ping.entries[u32(offset.tensor[curr_ParentRight].cols)]);

  if (gId.x >= parentLeft_m || gId.y >= parentRight_n) {
    return; // Guard against out-of-bounds work group sizes
  }

  var result = f32(0);
  for (var i = 0u; i < parentLeft_n; i = i + 1u) {
    let a = parentLeft_DataFirstIndex + i + gId.x * parentLeft_n;
    let b = parentRight_DataFirstIndex +  gId.y + i * parentRight_n;
    result += ping.entries[a] * ping.entries[b];
  }
  let index = gId.y + gId.x * parentRight_n;
  ping.entries[curr_DataFirstIndex + index] = result;

  // clean up the gradient data
  let curr_GradientData = u32(offset.tensor[t].gradientData);
  ping.entries[curr_GradientData + index] = f32(0);
}