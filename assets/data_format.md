# Data Format 

```bash
<root-directory>/
    config.json
    prompts.json # a list of all considered prompts
    comparisons.json # (optional) a list of all games already played
    gpt_prompts/
    methods/
        <method-name-1>/
            - <prompt-id-1>/ # Index from 0, using prompts.json
                -<seed-1>/ # Currently using single seed (0) only
                    - rgb_0.png
                    ...
                    - rgb_119.png
                    - normal_0.png
                    ...
                    - normal_119.png
                -<seed-2>
                ...
                -<seed-n>
            ...
            - <prompt-id-m>/
        - <method-name-2>/
        ...
        - <method-name-n>/
```

```json
// config.json
{
  // An ordered list of criteria
  // This is ordered so that we know how to parse the GPT response
  "criteria": [
    (<criteria-name-1>, <criteria-description-1>)
    ...,
    (<criteria-name-k>: <criteria-description-k>)
  ],
  // How we ensemble different types of questions.
  "ensembles": [
    {
      "num_views": ...,
      "rgb": ...,
      "normal": ...,
      "gpt_prompt": ...,
      "dimensions": [...],
      "num_comparisons": ...
    },
    ...
  ],
  // A default set of prompts if not specified in ensembles
  "gpt_prompts": [
        <file_path_to_the_prompt>, ...
    ],
  // Existing elo scores
  "scores": {
    // criteria id -> method_name -> score
    0: {  
        <method_name_1>: <score_1>,
        ...
       },
    ...
  }
}
```

```json
// prompts.json
// Containing a list of prompt for the dataset
[
    "<text prompt 1>",
    ...,
    "<text prompt m>"
]
```

```json
// comparisons.json
// Containing a list of existing GPT-4V comparison results.
// Result: 1 = left is better, 2 = right is better, 3 = cannot distinguish
{
    <criteria-name-1>: [
        {
            "m1": <method-id-1>, 
            "m2": <method-id-2>, 
            "prompt": <prompt>,
            "result": <result> # -1: m1 wins, 1: m2 wins, 0: draw
        },
        ...
    ],
    ...,
    <criteria-name-k>: [
        {
            "m1": <method-id-1>, 
            "m2": <method-id-2>, 
            "prompt": <prompt>,
            "result": <result> # -1: m1 wins, 1: m2 wins, 0: draw
        },
        ...
    ],
}
```
