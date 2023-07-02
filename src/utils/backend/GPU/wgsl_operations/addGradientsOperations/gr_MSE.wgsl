fn gr_MSE(gId : vec3u, currTensor : u32, childTensor: u32){
    let curr_m = u32(ping.entries[u32(offset.tensor[currTensor].rows)]);    
    let curr_n = u32(ping.entries[u32(offset.tensor[currTensor].cols)]);

    if (gId.x >= curr_m || gId.y >= curr_n) {
        return; // Guard against out-of-bounds work group sizes
    } 

    let curr_partner_id = u32(ping.entries[u32(offset.tensor[currTensor].partner_id)]);
    let curr_partner_data = u32(ping.entries[u32(offset.tensor[curr_partner_id].data)]);

    let curr_Data = u32(offset.tensor[currTensor].data);
    let curr_GradientData = u32(offset.tensor[currTensor].gradientData);

    let index = gId.y + gId.x * curr_n;
    ping.entries[curr_GradientData + index] = (f32(2.0) / f32(curr_m)) * (ping.entries[curr_Data + index] - ping.entries[curr_partner_data + index]);
}