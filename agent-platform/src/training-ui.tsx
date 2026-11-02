import {createContext,useContext,useEffect,useState} from 'react';
import type {ReactNode} from 'react';
import {getData} from './api';
import {vehicleLabel} from './vehicles';
import {ResourceTable,cellView} from './ui';
import type {ComponentProps} from 'react';
import './training-ui.css';
export const trainingStatus=(v:unknown)=>({assigned:'已分配',in_progress:'学习中',submitted:'待验收',reviewed:'已验收',returned:'退回补学',observation:'持续观察',observe:'持续观察',interview:'待面谈',approved:'已审核',pending_review:'待审核',draft:'草稿',disabled:'已停用',demo_draft_requires_team_review:'待审核',high:'高',medium:'中',low:'低'} as Record<string,string>)[String(v)]||String(v||'—');
const Context=createContext<{course:(v:unknown)=>string;driver:(v:unknown)=>string}>({course:(_v:unknown)=>'课程信息待加载',driver:(_v:unknown)=>'司机信息待加载'});
export const useTrainingNames=()=>useContext(Context);
export function TrainingSurface({children}:{children:ReactNode}){const [courses,setCourses]=useState<Record<string,string>>({}),[drivers,setDrivers]=useState<Record<string,string>>({});useEffect(()=>{let active=true;async function load(resource:string){const rows:Record<string,unknown>[]=[];for(let offset=0;;offset+=200){const r=await getData<Record<string,unknown>>(resource,{limit:200,offset});rows.push(...r.rows);if(r.rows.length<200)break}return rows}void load('admin.training_courses').then(rows=>{if(active)setCourses(Object.fromEntries(rows.map(r=>[String(r.course_id),String(r.title||'未命名课程')])))}).catch(()=>{});void load('admin.drivers').then(rows=>{if(active)setDrivers(Object.fromEntries(rows.map(r=>[String(r.driver_id),vehicleLabel(r.gpsno)+'司机'])))}).catch(()=>{});return()=>{active=false}},[]);return <Context.Provider value={{course:v=>courses[String(v)]||'课程信息待加载',driver:v=>drivers[String(v)]||'司机信息待加载'}}><div className="training-surface">{children}</div></Context.Provider>}
export function TrainingTable(props:ComponentProps<typeof ResourceTable>){const names=useTrainingNames();return <ResourceTable {...props} renderCell={(key,value)=>key==='task_id'?'培训任务':key==='driver_id'?names.driver(value):key==='course_id'?names.course(value):key==='status'?trainingStatus(value):key==='mandatory'?(Number(value)?'必修':'选修'):key.endsWith('_at')&&value?new Date(String(value)).toLocaleDateString('zh-CN'):cellView(key,value)}/>}
