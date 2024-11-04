
        uniform sampler2D images[1];
        varying highp vec2 texCoord;
        uniform highp vec2 inputTexel;
        uniform highp vec2 outputTexel;

        highp vec4 fetch(sampler2D image, highp float x, highp float y) {
            return texture2D(image, vec2(x, y));
        }

        void main() {
            highp vec2 inputLoc = ((texCoord / outputTexel) + vec2(1.0, 1.0))*inputTexel;

            highp float
                x0 = inputLoc.x - inputTexel.x,
                x1 = inputLoc.x,

                y0 = inputLoc.y - inputTexel.y,
                y1 = inputLoc.y;

            highp vec4 i0, i1, i2, i3;
            
            i0 = fetch(images[0], x0, y0);
            i1 = fetch(images[0], x1, y0);
            i2 = fetch(images[0], x0, y1);
            i3 = fetch(images[0], x1, y1);

            gl_FragColor = max(max(i0, i1), max(i2, i3));
        }
    