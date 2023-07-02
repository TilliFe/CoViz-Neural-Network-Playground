fn cleanupGradients(gId : vec3u, t : u32){
    let curr_m = u32(ping.entries[u32(offset.tensor[t].rows)]);   
    let curr_n = u32(ping.entries[u32(offset.tensor[t].cols)]);
    let tensorType = u32(ping.entries[u32(offset.tensor[t].types)]);

    let curr_Data = u32(offset.tensor[t].data);
    let curr_GradientData = u32(offset.tensor[t].gradientData);

    if (gId.x >= curr_m || gId.y >= curr_n) {
        return; // Guard against out-of-bounds work group sizes or not required update of the data entries
    }

    let index = gId.x * curr_n + gId.y;
    ping.entries[curr_GradientData + index] = f32(0);

    if(tensorType != u32(0)){
        ping.entries[curr_Data + index] = f32(0);
    }
}