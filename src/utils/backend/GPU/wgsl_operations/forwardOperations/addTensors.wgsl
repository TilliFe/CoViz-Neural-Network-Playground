fn addTensors(gId : vec3u, t : u32){
    let curr_m = u32(ping.entries[u32(offset.tensor[t].rows)]);
    let curr_n = u32(ping.entries[u32(offset.tensor[t].cols)]);
    let curr_DataFirstIndex = u32(offset.tensor[t].data);
    let curr_ParentLeft = u32(ping.entries[u32(offset.tensor[t].parents)]);
    let curr_ParentRight = u32(ping.entries[u32(offset.tensor[t].parents) + 1u]);
    let parentLeft_DataFirstIndex = u32(offset.tensor[curr_ParentLeft].data);
    let parentRight_DataFirstIndex = u32(offset.tensor[curr_ParentRight].data);
    let curr_GradientData = u32(offset.tensor[t].gradientData);

    if (gId.x >= curr_m || gId.y >= curr_n) {
        return; // Guard against out-of-bounds work group sizes
    }

    let index = gId.y + gId.x * curr_n;
    ping.entries[curr_DataFirstIndex + index] = ping.entries[parentLeft_DataFirstIndex + index] + ping.entries[parentRight_DataFirstIndex + index];

    // clean up the gradient data    
    // ping.entries[curr_GradientData + index] = f32(0.0);

}