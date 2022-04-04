using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Charge_Calculator
{
    public partial class Form1
    {
        int totalDuration = 0;

        private void GenerateTableSpecific()
        {
            r = new Random();
            s = new Random();
            TimeSpan start, end;
            int maxSeconds, seconds;
            List<TimeSpan> tsList;

            if (dgvResults.Rows.Count > 0)
                dgvResults.Rows.Clear();

            start = TimeSpan.FromHours(8);
            end = TimeSpan.FromHours(24);
            maxSeconds = (int)((end - start).TotalSeconds);
            tsList = new List<TimeSpan>();

            for (int i = 0; i < parkingEventsSpecific.Length * 2; i++)
            {
                Debug.WriteLine(parkingEventsSpecific.Length);
                seconds = s.Next(maxSeconds);
                tsList.Add(start.Add(TimeSpan.FromSeconds(seconds)));
            }

            tsList.Sort();
            totalDuration = 0;

            //MINUTE IS ALWAYS ROUNDED DOWN SO IF I HAVE DURATION OF 234.23 MINUTES IT BECOMES 234 MINUTES. THIS MEANS MY SECONDS COUNT WILL ALWAYS BE ZERO. MAWAKEF CAN CHANGE THIS IF THEY WANT
            for (int i = 0, listIndex = 0; i < parkingEventsSpecific.Length; i++, listIndex += 2)
            {
                dgvResults.Rows.Add(i + 1 + "", tsList[listIndex], tsList[listIndex + 1], "", parkingEventsSpecific[i] + "", (int)(tsList[listIndex + 1] - tsList[listIndex]).TotalMinutes, "", "");
                totalDuration += (int)(tsList[listIndex + 1] - tsList[listIndex]).TotalMinutes;
            }

            dgvResults.ClearSelection();
        }

        private void btnCalculateSpecific_Click(object sender, EventArgs e)
        {
            richTextBox1.Clear();

            string currentParkingType, tempParkingType;
            int hours, temp, pHours = 0, pMinutes = 0, sHours = 0, sMinutes = 0, duration, currentDuration;
            TimeSpan currentStart, currentEnd, tempStart, expectedEnd;
            DataGridViewCell dgvc;

            if (parkingEventsSpecific.Length > 0)
            {
                GenerateTableSpecific();

                //first row doesn't need a loop
                duration = (int)(dgvResults.Rows[0].Cells[5].Value);

                if (dgvResults.Rows[0].Cells[4].Value.ToString().CompareTo("P") == 0)
                {
                    if (duration / 60 == 0)
                        pMinutes = duration;

                    else
                    {
                        pHours = duration / 60;
                        pMinutes = duration % 60;
                    }

                    dgvc = dgvResults.Rows[0].Cells[8];
                    dgvc.Value = pHours + "";
                    dgvc = dgvResults.Rows[0].Cells[9];
                    dgvc.Value = pMinutes + "";
                }

                else
                {
                    if (duration / 60 == 0)
                        sMinutes = duration;

                    else
                    {
                        sHours = duration / 60;
                        sMinutes = duration % 60;
                    }

                    dgvc = dgvResults.Rows[0].Cells[10];
                    dgvc.Value = sHours + "";
                    dgvc = dgvResults.Rows[0].Cells[11];
                    dgvc.Value = sMinutes + "";
                }


                if (parkingEventsSpecific.Length > 1)
                {
                    //find which time range of the above rows include current parking start time (which start and expected end of above rows include the time of current parking)
                    for (int currentRowIndex = 1, startFromRow = 0; currentRowIndex < parkingEventsSpecific.Length; currentRowIndex++)
                    {
                        currentStart = TimeSpan.Parse(dgvResults.Rows[currentRowIndex].Cells[1].Value.ToString());
                        currentEnd = TimeSpan.Parse(dgvResults.Rows[currentRowIndex].Cells[2].Value.ToString());
                        currentParkingType = dgvResults.Rows[currentRowIndex].Cells[4].Value.ToString();
                        currentDuration = (int)dgvResults.Rows[currentRowIndex].Cells[5].Value;

                        hours = (int)(dgvResults.Rows[currentRowIndex].Cells[5].Value) / 60 + 1;
                        expectedEnd = currentStart.Add(TimeSpan.FromHours(hours));
                        dgvc = dgvResults.Rows[currentRowIndex].Cells[3];
                        dgvc.Value = expectedEnd;
                        dgvc = dgvResults.Rows[currentRowIndex].Cells[7];
                        dgvc.Value = startFromRow + "";

                        for (int j = startFromRow; j < currentRowIndex; j++)
                        {
                            tempStart = TimeSpan.Parse(dgvResults.Rows[j].Cells[1].Value.ToString());
                            hours = (int)(dgvResults.Rows[j].Cells[5].Value) / 60 + 1;
                            expectedEnd = tempStart.Add(TimeSpan.FromHours(hours));
                            dgvc = dgvResults.Rows[j].Cells[3];
                            dgvc.Value = expectedEnd;

                            tempParkingType = dgvResults.Rows[j].Cells[4].Value.ToString();

                            if (currentStart < expectedEnd && currentEnd > tempStart && currentEnd <= expectedEnd)
                            {
                                dgvc = dgvResults.Rows[j + 1].Cells[6];
                                dgvc.Value = "Within range";
                                //changeStartFromRow = true;

                                if (currentParkingType.CompareTo("P") == 0 && tempParkingType.CompareTo("P") == 0)
                                {
                                    if (sMinutes > 0)
                                    {
                                        sMinutes = 0;
                                        sHours++;
                                    }

                                    temp = pMinutes + currentDuration;

                                    if (temp >= 60)
                                    {
                                        pHours += temp / 60;
                                        pMinutes = temp % 60;
                                    }

                                    else
                                        pMinutes = temp;
                                }

                                else if (currentParkingType.CompareTo("P") == 0 && tempParkingType.CompareTo("S") == 0)
                                {
                                    if (sMinutes > 0)
                                    {
                                        sMinutes = 0;
                                        sHours++;
                                    }

                                    pHours = currentDuration / 60;
                                    pMinutes = currentDuration % 60;
                                }

                                else if (currentParkingType.CompareTo("S") == 0 && tempParkingType.CompareTo("S") == 0)
                                {
                                    if (pMinutes > 0)
                                    {
                                        pMinutes = 0;
                                        pHours++;
                                    }

                                    temp = sMinutes + currentDuration;

                                    if (temp >= 60)
                                    {
                                        sHours += temp / 60;
                                        sMinutes = temp % 60;
                                    }

                                    else
                                        sMinutes = temp;
                                }

                                else if (currentParkingType.CompareTo("S") == 0 && tempParkingType.CompareTo("P") == 0)
                                {
                                    btnCalculateSpecific.Enabled = false;
                                    btnCalculateSpecific.Enabled = true;
                                    temp = pMinutes + currentDuration;

                                    if (temp >= 60)
                                    {
                                        pMinutes = 0;
                                        pHours += temp / 60;
                                        sMinutes += temp % 60;

                                        if (sMinutes >= 60)
                                        {
                                            sHours += sMinutes / 60;
                                            sMinutes %= 60;
                                        }
                                    }

                                    else
                                        pMinutes = temp;
                                }
                            }

                            else if (currentStart < expectedEnd && currentEnd > expectedEnd)
                            {
                                dgvc = dgvResults.Rows[j + 1].Cells[6];
                                dgvc.Value = "start yes end no";
                                //changeStartFromRow = true;

                                if (currentParkingType.CompareTo("P") == 0 && tempParkingType.CompareTo("P") == 0)
                                {
                                    if (sMinutes > 0)
                                    {
                                        sMinutes = 0;
                                        sHours++;
                                    }

                                    if (pMinutes > 0)
                                    {
                                        pHours++;
                                        pMinutes = 0;
                                    }

                                    pMinutes = (int)(currentDuration - (expectedEnd - currentStart).TotalMinutes);
                                    if (pMinutes >= 60)
                                    {
                                        pHours += pMinutes / 60;
                                        pMinutes %= 60;
                                    }
                                }

                                else if (currentParkingType.CompareTo("P") == 0 && tempParkingType.CompareTo("S") == 0)
                                {
                                    if (sMinutes > 0)
                                    {
                                        sMinutes = 0;
                                        sHours++;
                                    }

                                    pHours += currentDuration / 60;
                                    pMinutes += currentDuration % 60;

                                    if (pMinutes > 60)
                                        pHours++;
                                }

                                else if (currentParkingType.CompareTo("S") == 0 && tempParkingType.CompareTo("S") == 0)
                                {
                                    if (pMinutes > 0)
                                    {
                                        pMinutes = 0;
                                        pHours++;
                                    }

                                    if (sMinutes > 0)
                                    {
                                        sHours++;
                                        sMinutes = 0;
                                    }

                                    sMinutes = (int)(currentDuration - (expectedEnd - currentStart).TotalMinutes);
                                    if (sMinutes >= 60)
                                    {
                                        sHours += sMinutes / 60;
                                        sMinutes %= 60;
                                    }
                                }

                                else if (currentParkingType.CompareTo("S") == 0 && tempParkingType.CompareTo("P") == 0)
                                {
                                    pHours++;
                                    pMinutes = 0;

                                    sMinutes = (int)(currentDuration - (expectedEnd - currentStart).TotalMinutes);
                                    if (sMinutes >= 60)
                                    {
                                        sHours += sMinutes / 60;
                                        sMinutes %= 60;
                                    }
                                }
                            }

                            else if (currentStart > expectedEnd && currentEnd > expectedEnd)
                            {
                                dgvc = dgvResults.Rows[j + 1].Cells[6];
                                dgvc.Value = "Outside of range";
                                //changeStartFromRow = true;

                                if (currentParkingType.CompareTo("P") == 0 && tempParkingType.CompareTo("P") == 0)
                                {
                                    if (sMinutes > 0)
                                    {
                                        sMinutes = 0;
                                        sHours++;
                                    }

                                    pHours++;
                                    pMinutes = currentDuration;

                                    if (pMinutes >= 60)
                                    {
                                        pHours += pMinutes / 60;
                                        pMinutes %= 60;
                                    }
                                }

                                else if (currentParkingType.CompareTo("P") == 0 && tempParkingType.CompareTo("S") == 0)
                                {
                                    if (sMinutes > 0)
                                    {
                                        sMinutes = 0;
                                        sHours++;
                                    }

                                    if (pMinutes > 0)
                                    {
                                        pHours++;
                                        pMinutes = 0;
                                    }

                                    pHours += currentDuration / 60;
                                    pMinutes = currentDuration % 60;
                                }

                                else if (currentParkingType.CompareTo("S") == 0 && tempParkingType.CompareTo("S") == 0)
                                {
                                    if (pMinutes > 0)
                                    {
                                        pMinutes = 0;
                                        pHours++;
                                    }

                                    sHours++;
                                    sMinutes = currentDuration;

                                    if (sMinutes >= 60)
                                    {
                                        sHours += sMinutes / 60;
                                        sMinutes %= 60;
                                    }
                                }

                                else if (currentParkingType.CompareTo("S") == 0 && tempParkingType.CompareTo("P") == 0)
                                {
                                    if (pMinutes > 0)
                                    {
                                        pHours++;
                                        pMinutes = 0;
                                    }

                                    sMinutes = currentDuration;
                                    if (sMinutes >= 60)
                                    {
                                        sHours += sMinutes / 60;
                                        sMinutes %= 60;
                                    }
                                }
                            }

                            startFromRow++;

                            dgvc = dgvResults.Rows[currentRowIndex].Cells[8];
                            dgvc.Value = pHours + "";
                            dgvc = dgvResults.Rows[currentRowIndex].Cells[9];
                            dgvc.Value = pMinutes + "";
                            dgvc = dgvResults.Rows[currentRowIndex].Cells[10];
                            dgvc.Value = sHours + "";
                            dgvc = dgvResults.Rows[currentRowIndex].Cells[11];
                            dgvc.Value = sMinutes + "";
                        }
                    }

                    if (pMinutes > 0)
                    {
                        pHours++;
                        pMinutes = 0;
                    }

                    if (sMinutes > 0)
                    {
                        sHours++;
                        sMinutes = 0;
                    }

                    richTextBox1.AppendText("Total P Hours: " + pHours
                        + "\nTotal S Hours: " + sHours
                        + "\nTotal Hours" + (pHours + sHours)
                        + "\nTotal Duration: " + ((int)(totalDuration / 60)) + " hrs " + (totalDuration % 60) + " mins");
                }
            }
        }
    }
}


/*

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Charge_Calculator
{
    public partial class Form1
    {
        private void btnCalculate_Click(object sender, EventArgs e)
        {
            if (parkingEvents.Length > 0)
            {
                r = new Random();
                s = new Random();
                TimeSpan start, end;
                int maxSeconds, seconds;
                List<TimeSpan> tsList;

                if (dgvResults.Rows.Count > 0)
                    dgvResults.Rows.Clear();

                start = TimeSpan.FromHours(8);
                end = TimeSpan.FromHours(24);
                maxSeconds = (int)((end - start).TotalSeconds);
                tsList = new List<TimeSpan>();

                for (int i = 0; i < parkingEvents.Length * 2; i++)
                {
                    seconds = s.Next(maxSeconds);
                    tsList.Add(start.Add(TimeSpan.FromSeconds(seconds)));
                }

                tsList.Sort();

                for (int i = 0, listIndex = 0; i < parkingEvents.Length; i++, listIndex += 2)
                    dgvResults.Rows.Add(tsList[listIndex], tsList[listIndex + 1], parkingEvents[i] + "", (int)(tsList[listIndex + 1] - tsList[listIndex]).TotalMinutes); // w/o int, i'd get x.y so i'm rounding it down from x.y to x instead of rounding it up from x.y to z

                dgvResults.ClearSelection();

                string previousParking = "", currentParking;
                int pHours = 0, pMinutes = 0, sHours = 0, sMinutes = 0, duration;

                for (int i = 0; i < parkingEvents.Length; i++)
                {
                    currentParking = dgvResults.Rows[i].Cells[2].Value.ToString();
                    duration = int.Parse(dgvResults.Rows[i].Cells[3].Value.ToString());

                    if (currentParking.CompareTo("P") == 0)
                    {
                        if (previousParking.CompareTo("P") == 0)
                        {
                            pMinutes += duration;
                            if (pMinutes >= 60)
                            {
                                pHours = pMinutes / 60 + pHours;
                                pMinutes %= 60;
                            }
                        }

                        else if (previousParking.CompareTo("S") == 0)
                        {

                        }

                        else
                        {
                            if (duration / 60 == 0)
                                pMinutes = duration;
                            else
                            {
                                pMinutes = duration;
                                pHours = pMinutes / 60 + pHours;
                                pMinutes -= (pHours * 60);
                            }
                        }
                    }

                    else if (currentParking.CompareTo("S") == 0)
                    {
                        if (previousParking.CompareTo("S") == 0)
                        {
                            sMinutes += duration;
                            if (sMinutes >= 60)
                            {
                                sHours = sMinutes / 60 + sHours;
                                sMinutes %= 60;
                            }
                        }

                        else if (previousParking.CompareTo("P") == 0)
                        {
                            pMinutes += duration;
                            if (pMinutes >= 60)
                            {
                                pHours = pMinutes / 60 + pHours;
                                sMinutes = pMinutes % 60;
                                pMinutes = 0;
                            }
                        }

                        else
                        {
                            if (duration / 60 == 0)
                                sMinutes = duration;
                            else
                            {
                                sMinutes = duration;
                                sHours = sMinutes / 60 + sHours;
                                sMinutes -= (sHours * 60);
                            }
                        }
                    }

                    previousParking = currentParking;
                }

                dgvResults.Rows.Add("P: " + pHours + " hr  " + pMinutes + " m", "S: " + sHours + " hr " + sMinutes + " m");
            }
        }
    }
}
 */

/*
 using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Charge_Calculator
{
    public partial class Form1
    {
        private void GenerateTable()
        {
            r = new Random();
            s = new Random();
            TimeSpan start, end;
            int maxSeconds, seconds;
            List<TimeSpan> tsList;

            if (dgvResults.Rows.Count > 0)
                dgvResults.Rows.Clear();

            start = TimeSpan.FromHours(8);
            end = TimeSpan.FromHours(24);
            maxSeconds = (int)((end - start).TotalSeconds);
            tsList = new List<TimeSpan>();

            for (int i = 0; i < parkingEvents.Length * 2; i++)
            {
                seconds = s.Next(maxSeconds);
                tsList.Add(start.Add(TimeSpan.FromSeconds(seconds)));
            }

            tsList.Sort();

            //MINUTE IS ALWAYS ROUNDED DOWN SO IF I HAVE DURATION OF 234.23 MINUTES IT BECOMES 234 MINUTES. THIS MEANS MY SECONDS COUNT WILL ALWAYS BE ZERO. MAWAKEF CAN CHANGE THIS IF THEY WANT
            for (int i = 0, listIndex = 0; i < parkingEvents.Length; i++, listIndex += 2)
                dgvResults.Rows.Add(i+1 + "", tsList[listIndex], tsList[listIndex + 1], parkingEvents[i] + "", (int)(tsList[listIndex + 1] - tsList[listIndex]).TotalMinutes, "");

            dgvResults.ClearSelection();
        }

        private void btnCalculate_Click(object sender, EventArgs e)
        {
            string currentParkingType, prevParkingType, tempParkingType;
            int temp, pHours = 0, pMinutes = 0, sHours = 0, sMinutes = 0, duration, currentDuration;
            TimeSpan currentStart, currentEnd, tempStart, tempEnd, expectedEnd;
            DataGridViewCell dgvc;

            if (parkingEvents.Length > 0)
            {
                GenerateTable();
                
                //first row doesn't need a loop
                duration = (int)(dgvResults.Rows[0].Cells[4].Value);

                if (dgvResults.Rows[0].Cells[3].Value.ToString().CompareTo("P") == 0)
                {
                    if (duration / 60 == 0)
                        pMinutes = duration;

                    else
                    {
                        pHours = duration / 60;
                        pMinutes = duration % 60;
                    }

                    dgvc = dgvResults.Rows[0].Cells[6];
                    dgvc.Value = pHours + "";
                    dgvc = dgvResults.Rows[0].Cells[7];
                    dgvc.Value = pMinutes + "";
                }

                else
                {
                    if (duration / 60 == 0)
                        sMinutes = duration;

                    else
                    {
                        sHours = duration / 60;
                        sMinutes = duration % 60;
                    }

                    dgvc = dgvResults.Rows[0].Cells[8];
                    dgvc.Value = sHours + "";
                    dgvc = dgvResults.Rows[0].Cells[9];
                    dgvc.Value = sMinutes + "";
                }


                if (parkingEvents.Length > 1)
                {
                    //find which time range of the above rows include current parking start time (which start and expected end of above rows include the time of current parking)
                    for (int currentRowIndex = 1, startFromRow = 0; currentRowIndex < parkingEvents.Length; currentRowIndex++)
                    {
                        currentStart = TimeSpan.Parse(dgvResults.Rows[currentRowIndex].Cells[1].Value.ToString());
                        currentEnd = TimeSpan.Parse(dgvResults.Rows[currentRowIndex].Cells[2].Value.ToString());
                        currentParkingType = dgvResults.Rows[currentRowIndex].Cells[3].Value.ToString();
                        currentDuration = (int)dgvResults.Rows[currentRowIndex].Cells[4].Value;

                        dgvc = dgvResults.Rows[currentRowIndex].Cells[5];
                        dgvc.Value = startFromRow + "";

                        for (int j = startFromRow; j < currentRowIndex; j++)
                        {
                            tempStart = TimeSpan.Parse(dgvResults.Rows[j].Cells[1].Value.ToString());
                            tempEnd = TimeSpan.Parse(dgvResults.Rows[j].Cells[2].Value.ToString());
                            expectedEnd = TimeSpan.FromHours((int)tempEnd.TotalMinutes / 60 + 1);
                            tempParkingType = dgvResults.Rows[j].Cells[3].Value.ToString();

                            if (currentStart < expectedEnd && currentEnd > tempStart && currentEnd <= expectedEnd)
                            {
                                Debug.WriteLine("start and end within range");
                                //changeStartFromRow = true;

                                if (currentParkingType.CompareTo("P") == 0 && tempParkingType.CompareTo("P") == 0)
                                {
                                    temp = pMinutes + currentDuration;

                                    if (temp >= 60)
                                    {
                                        pHours += temp / 60;
                                        pMinutes = temp % 60;
                                    }

                                    else
                                        pMinutes = temp;
                                }

                                else if (currentParkingType.CompareTo("P") == 0 && tempParkingType.CompareTo("S") == 0)
                                {
                                    
                                }

                                else if (currentParkingType.CompareTo("S") == 0 && tempParkingType.CompareTo("S") == 0)
                                {
                                    temp = sMinutes + currentDuration;

                                    if (temp >= 60)
                                    {
                                        sHours += temp / 60;
                                        sMinutes = temp % 60;
                                    }

                                    else
                                        sMinutes = temp;
                                }

                                else if (currentParkingType.CompareTo("S") == 0 && tempParkingType.CompareTo("P") == 0)
                                {
                                    temp = pMinutes + currentDuration;

                                    if (temp >= 60)
                                    {
                                        pHours += temp / 60;
                                        pMinutes = 0;
                                        sMinutes = temp % 60;
                                    }

                                    else
                                        pMinutes = temp;
                                }
                            }

                            else if (currentStart < expectedEnd && currentEnd > expectedEnd)
                            {
                                Debug.WriteLine("start within range but end outside range");
                                //changeStartFromRow = true;

                                if (currentParkingType.CompareTo("P") == 0 && tempParkingType.CompareTo("P") == 0)
                                {
                                    pHours++;
                                    pMinutes += (int)(currentEnd - expectedEnd).TotalMinutes;

                                    if (pMinutes > 60)
                                    {
                                        pHours += pMinutes / 60;
                                        pMinutes %= 60;
                                    }
                                }

                                else if (currentParkingType.CompareTo("P") == 0 && tempParkingType.CompareTo("S") == 0)
                                {

                                }

                                else if (currentParkingType.CompareTo("S") == 0 && tempParkingType.CompareTo("S") == 0)
                                {
                                    sHours++;
                                    sMinutes += (int)(currentEnd - expectedEnd).TotalMinutes;

                                    if (pMinutes > 60)
                                    {
                                        pHours += pMinutes / 60;
                                        pMinutes %= 60;
                                    }
                                }

                                else if (currentParkingType.CompareTo("S") == 0 && tempParkingType.CompareTo("P") == 0)
                                {

                                }
                            }

                            else if (currentStart > expectedEnd && currentEnd > expectedEnd)
                            {
                                Debug.WriteLine("start and end outside of range");
                                //changeStartFromRow = true;

                                if (currentParkingType.CompareTo("P") == 0 && tempParkingType.CompareTo("P") == 0)
                                {
                                    pHours++;
                                    pMinutes = currentDuration;

                                    if (pMinutes > 60)
                                    {
                                        pHours += pMinutes / 60;
                                        pMinutes %= 60;
                                    }
                                }

                                else if (currentParkingType.CompareTo("P") == 0 && tempParkingType.CompareTo("S") == 0)
                                {

                                }

                                else if (currentParkingType.CompareTo("S") == 0 && tempParkingType.CompareTo("S") == 0)
                                {
                                    sHours++;
                                    sMinutes = currentDuration;

                                    if (sMinutes > 60)
                                    {
                                        sHours += pMinutes / 60;
                                        sMinutes %= 60;
                                    }
                                }

                                else if (currentParkingType.CompareTo("S") == 0 && tempParkingType.CompareTo("P") == 0)
                                {

                                }
                            }

                            /*if (changeStartFromRow == true)
                            {
                                changeStartFromRow = false;
                                startFromRow++;
                            }* /

startFromRow++;

dgvc = dgvResults.Rows[currentRowIndex].Cells[6];
dgvc.Value = pHours + "";
dgvc = dgvResults.Rows[currentRowIndex].Cells[7];
dgvc.Value = pMinutes + "";
dgvc = dgvResults.Rows[currentRowIndex].Cells[8];
dgvc.Value = sHours + "";
dgvc = dgvResults.Rows[currentRowIndex].Cells[9];
dgvc.Value = sMinutes + "";
                        }
                    }
                }
            }
        }
    }
} 
 */ 