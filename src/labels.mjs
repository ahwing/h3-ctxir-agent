// Reference-label map exactly as H3-Base (sglang minimax_h3 text_encoding.py / presentation.py) builds it.
// Walk conditions in request order:
//   image  -> <Picture i>
//   audio  -> <Audio j>
//   video  -> <Audio j> first IF the file has a soundtrack, then <Video k>
// Counters are per type. A raw user prompt that says "audio 1" for the first audio FILE is therefore wrong whenever
// a reference video with sound comes earlier. The agent gets this table so its IR binds labels the way the model reads them.

/**
 * @param {{type:'image'|'video'|'audio', file?:string}[]} conditions
 * @param {{hasAudio?:boolean}[]} probes same order as conditions (only video entries are read)
 * @returns {{label:string, type:string, conditionIndex:number, source:string}[]}
 */
export function computeLabels(conditions, probes = []) {
  const n = {image: 0, audio: 0, video: 0};
  const out = [];
  conditions.forEach((c, i) => {
    if (c.type === 'image') out.push({label: `<Picture ${++n.image}>`, type: 'image', conditionIndex: i, source: 'image file'});
    else if (c.type === 'audio') out.push({label: `<Audio ${++n.audio}>`, type: 'audio', conditionIndex: i, source: 'audio file'});
    else if (c.type === 'video') {
      if (probes[i]?.hasAudio) out.push({label: `<Audio ${++n.audio}>`, type: 'audio', conditionIndex: i, source: 'soundtrack of the video file'});
      out.push({label: `<Video ${++n.video}>`, type: 'video', conditionIndex: i, source: 'video file'});
    } else throw new Error(`unknown condition type ${c.type}`);
  });
  return out;
}

/** All asset labels (Picture/Video/Audio) present in a label map, as a Set of strings like "<Audio 2>". */
export const labelSet = (labels) => new Set(labels.map((l) => l.label));
